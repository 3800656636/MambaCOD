import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import timm
import argparse
from tqdm import tqdm
from mamba_ssm import Mamba


class SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.proj = nn.Conv2d(d_model, d_model, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W

        x1 = x.view(B, C, L).transpose(1, 2)
        x2 = x.transpose(2, 3).contiguous().view(B, C, L).transpose(1, 2)
        x3 = x.flip(dims=[3]).view(B, C, L).transpose(1, 2)
        x4 = x.flip(dims=[2]).view(B, C, L).transpose(1, 2)

        y1, y2, y3, y4 = self.mamba(x1), self.mamba(x2), self.mamba(x3), self.mamba(x4)

        y1 = y1.transpose(1, 2).view(B, C, H, W)
        y2 = y2.transpose(1, 2).view(B, C, W, H).transpose(2, 3)
        y3 = y3.transpose(1, 2).view(B, C, H, W).flip(dims=[3])
        y4 = y4.transpose(1, 2).view(B, C, H, W).flip(dims=[2])

        return self.proj(y1 + y2 + y3 + y4)


class FeatureExtractionModule(nn.Module):
    def __init__(self, model_type='pvt_v2_b2', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_type, pretrained=pretrained, features_only=True)
        raw_channels = self.backbone.feature_info.channels()
        self.target_channels = [256, 512, 1024, 2048]
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(raw_channels[i], self.target_channels[i], kernel_size=1) for i in range(4)
        ])

    def forward(self, x):
        features = self.backbone(x)
        return [self.proj_layers[i](f) for i, f in enumerate(features)]


class CSVSSM(nn.Module):
    def __init__(self, k, channels):
        super().__init__()
        self.k = k
        curr_c = channels[k - 1]

        if k == 1:
            in_c = channels[0] + channels[1]
        elif k == 4:
            in_c = channels[2] + channels[3]
        else:
            in_c = channels[k - 2] + channels[k - 1] + channels[k]

        self.pre_conv = nn.Conv2d(in_c, curr_c, kernel_size=1)
        self.ssm = SS2D(d_model=curr_c)

    def forward(self, E):
        idx = self.k - 1
        Ek = E[idx]
        size = Ek.shape[2:]

        if self.k == 1:
            E_hat = torch.cat([Ek, F.interpolate(E[1], size=size, mode='bilinear')], dim=1)
        elif self.k == 4:
            E_hat = torch.cat([F.interpolate(E[2], size=size, mode='bilinear'), Ek], dim=1)
        else:
            E_prev = F.interpolate(E[idx - 1], size=size, mode='bilinear')
            E_next = F.interpolate(E[idx + 1], size=size, mode='bilinear')
            E_hat = torch.cat([E_prev, Ek, E_next], dim=1)

        E_tilde = self.ssm(self.pre_conv(E_hat))
        return E_tilde + Ek


class FactorizedConv(nn.Module):
    def __init__(self, in_c, out_c, ks):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, (ks, 1), padding=(ks // 2, 0)),
            nn.Conv2d(out_c, out_c, (1, ks), padding=(0, ks // 2)),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class HVSSM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        mid_c = channels // 4
        self.reduce = nn.Conv2d(channels, mid_c, kernel_size=1)
        self.h_convs = nn.ModuleList([FactorizedConv(mid_c, mid_c, ks) for ks in [3, 5, 7, 9]])
        self.ssms = nn.ModuleList([SS2D(d_model=mid_c) for _ in range(4)])
        self.merge = nn.Conv2d(channels, channels, kernel_size=1)
        self.shortcut = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, Z):
        Z_red = self.reduce(Z)
        h_list = []
        for i in range(4):
            h_i = self.h_convs[i](Z_red)
            h_hat_i = self.ssms[i](h_i)
            h_list.append(h_hat_i)

        H_tilde = self.merge(torch.cat(h_list, dim=1))
        return self.shortcut(Z) + H_tilde


class MambaCOD(nn.Module):
    def __init__(self, backbone_type='pvt_v2_b2', pretrained=True):
        super().__init__()
        self.encoder = FeatureExtractionModule(backbone_type, pretrained)
        channels = [256, 512, 1024, 2048]
        self.cs_vssms = nn.ModuleList([CSVSSM(k=i + 1, channels=channels) for i in range(4)])
        self.hvssms = nn.ModuleList([HVSSM(channels=c) for c in channels])
        self.heads = nn.ModuleList([nn.Conv2d(c, 1, kernel_size=1) for c in channels])

    def forward(self, x):
        E = self.encoder(x)
        E_check = [self.cs_vssms[i](E) for i in range(4)]
        Z_prime = [self.hvssms[i](E_check[i]) for i in range(4)]
        return [self.heads[i](Z_prime[i]) for i in range(4)]


class StructureLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def _weighted_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()

    def forward(self, preds, target):
        loss = 0
        weights = [1.0, 0.8, 0.6, 0.4]
        for i in range(len(preds)):
            res_pred = F.interpolate(preds[i], size=target.shape[2:], mode='bilinear', align_corners=True)
            loss += weights[i] * self._weighted_loss(res_pred, target)
        return loss


class CODDataset(Dataset):
    def __init__(self, root, size=224):
        self.img_dir = os.path.join(root, 'Image')
        self.gt_dir = os.path.join(root, 'GT')
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(('.jpg', '.png'))])
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        name = self.images[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert('RGB')
        gt = Image.open(os.path.join(self.gt_dir, os.path.splitext(name)[0] + '.png')).convert('L')
        return self.transform(img), self.gt_transform(gt)

    def __len__(self):
        return len(self.images)


def train(args):
    device = torch.device(args.device)
    loader = DataLoader(CODDataset(args.data_root, args.size), batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = MambaCOD(backbone_type=args.backbone).to(device)
    criterion = StructureLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}")
        for imgs, gts in pbar:
            imgs, gts = imgs.to(device), gts.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, gts)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if (epoch + 1) % args.save_step == 0:
            torch.save(model.state_dict(), f"mambacod_ep{epoch + 1}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="pvt_v2_b2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--save_step", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    train(parser.parse_args())