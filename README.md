# Position DETR

A PyTorch implementation for object detection based on the DETR architecture.

## 🧩 Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/wenkaizhao/Position-DETR.git
cd Position-DETR
```

### 2. Install Dependencies
```bash
# Install PyTorch (choose version according to your CUDA setup)
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

# Install other requirements
pip install -r requirements.txt
```

## ⚙️ Configuration

Edit `configs/train_config.py` before training to set:

```python
# Example configuration
coco_path = "/path/to/your/coco/"
num_epochs = 12
batch_size = 2
learning_rate = 1e-4
model_path = "configs/position_detr/position_detr_resnet50.py"
```

## 🚀 Run

### Training
```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch main.py
```

### Inference
```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch inference.py --coco-path /path/to/coco --model-config /path/to/model.py --checkpoint /path/to/checkpoint.pth
```
