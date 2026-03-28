# MambaCOD (Unofficial PyTorch Implementation)

This is an unofficial PyTorch reproduction of the paper:  
**"MambaCOD: Camouflaged object detection with state-space model"**  
*Published in Neurocomputing (2025)*  
Authors: Zhouyong Liu, Taotao Ji, Chunguo Li, Yongming Huang, Luxi Yang (School of Information Science and Engineering, Southeast University, China)

---

## 🌟 Introduction
This repository provides a complete implementation of **MambaCOD**, a cutting-edge framework for Camouflaged Object Detection (COD) based on the **State-Space Model (Mamba)** mechanism. 

### Key Features:
- **PVT-v2 Backbone**: High-efficiency feature extraction.
- **CS-VSSM (Cross-Scale Vision State-Space Module)**: Captures long-range contextual dependencies across different scales using 2D-selective scanning.
- **HVSSM (Hierarchical Vision State-Space Module)**: Enhances local and global receptive fields using factorized convolutions and Mamba blocks.
- **Weighted Pyramid Supervision**: Combined Weighted BCE and Weighted IoU loss for precise segmentation.

---

## 🏗️ Architecture Overview
The model follows the architecture described in the paper:
1. **Feature Extraction**: Multi-level features from PVT-v2 ($E_1, E_2, E_3, E_4$).
2. **CS-VSSM**: Aligns cross-scale features and performs 4-directional sequential scanning.
3. **HVSSM**: Hierarchical feature transformation with kernel sizes 3, 5, 7, and 9.
4. **Prediction Heads**: Multi-scale outputs for refined object boundaries.

---

## 🛠️ Installation

### Environment Requirements
- Python 3.10+
- PyTorch 2.1+ 
- CUDA 11.8+ (Required for Mamba kernels)

### Setup
```bash
# Clone the repository
git clone https://github.com/YourUsername/MambaCOD-Reproduction.git
cd MambaCOD-Reproduction

# Install core dependencies
pip install torch torchvision torchaudio
pip install timm tqdm Pillow

# Install Mamba-specific operators (Critical!)
pip install causal-conv1d>=1.2.0
pip install mamba-ssm
```

---

## 📂 Data Preparation
Please organize your COD datasets (e.g., **COD10K**, **CAMO**, **NC4K**) as follows:
```text
dataset_root/
├── Image/
│   ├── image_01.jpg
│   └── ...
└── GT/
    ├── image_01.png
    └── ...
```

---

## 🚀 Usage

### Training
To train MambaCOD on your dataset:
```bash
python main.py \
    --data_root ./data/COD10K/TrainDataset \
    --backbone pvt_v2_b2 \
    --batch_size 8 \
    --lr 1e-4 \
    --epochs 100 \
    --size 224
```

### Inference
*Coming soon / provided in `inference.py`*

---

## 📊 Reproduction Notes
- **Channel Scaling**: Following Eq.(10), channels are set to $[256, 512, 1024, 2048]$.
- **Mamba Scan**: Implemented 4-direction (L-to-R, T-to-B, R-to-L, B-to-T) selective scanning in `SS2D` module.
- **VRAM Usage**: Due to high-dimensional feature maps (2048-dim at Stage 4), a GPU with at least 24GB VRAM (e.g., RTX 3090/4090) is recommended for default settings.

---

## 📜 Citation
If you find this reproduction helpful, please cite the original paper:
```bibtex
@article{LIU2025MambaCOD,
  title={MambaCOD: Camouflaged object detection with state-space model},
  author={Zhouyong Liu and Taotao Ji and Chunguo Li and Yongming Huang and Luxi Yang},
  journal={Neurocomputing},
  volume={652},
  pages={131043},
  year={2025},
  publisher={Elsevier}
}
```

## ⚖️ Disclaimer
This is an **unofficial** reproduction. All rights to the algorithm belong to the original authors. Since no official code or weights were provided, performance may slightly differ from the reported values due to hyperparameter settings.
