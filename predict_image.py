"""
predict_image.py — Tamil Character Recognition from Image Files
================================================================
Usage:
    python predict_image.py your_image.jpg

Supports: .jpg, .jpeg, .png, .bmp, .tiff, .webp
The image should contain a single handwritten Tamil character.
Best results: dark ink on a light/white background.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps, ImageFilter

# ──────────────────────────────────────────────
# 1. NEURAL NETWORK ARCHITECTURE (matches training)
# ──────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.pool  = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x

class TamilNet(nn.Module):
    def __init__(self, num_classes=156):
        super().__init__()
        self.block1 = ConvBlock(1, 16)
        self.block2 = ConvBlock(16, 32)
        self.block3 = ConvBlock(32, 64)
        self.fc1 = nn.Linear(64 * 8 * 8, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x


# ──────────────────────────────────────────────
# 2. IMAGE PREPROCESSING (matches training pipeline)
# ──────────────────────────────────────────────

def preprocess_image(image_path):
    """
    Converts any photo of a handwritten Tamil character into
    the exact format the model expects (64x64, grayscale, normalised).
    """
    # Open and convert to grayscale
    img = Image.open(image_path).convert("L")

    # ---- Auto-detect if image is dark-on-light (photo) or light-on-dark (canvas) ----
    arr = np.array(img)
    mean_brightness = arr.mean()

    if mean_brightness > 128:
        # Photo: dark ink on white paper → invert so character is WHITE on BLACK
        img = ImageOps.invert(img)
        arr = np.array(img)

    # ---- Threshold to clean up noise (Otsu-like simple threshold) ----
    threshold = arr.max() * 0.3
    arr = np.where(arr > threshold, arr, 0).astype(np.uint8)
    img = Image.fromarray(arr)

    # ---- Slight thickening (dilation) to match training augmentation ----
    img = img.filter(ImageFilter.MaxFilter(3))

    # ---- Resize: make the longer side 48px, keep aspect ratio, Lanczos ----
    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError("Image appears to be blank or empty.")

    if w > h:
        new_w, new_h = 48, int(48 * h / w)
    else:
        new_w, new_h = int(48 * w / h), 48

    img = img.resize((new_w, new_h), Image.LANCZOS)

    # ---- Centre of mass centering on a 64x64 canvas ----
    canvas = Image.new("L", (64, 64), 0)
    arr = np.array(img, dtype=np.float32)

    # Find centre of mass
    total = arr.sum()
    if total == 0:
        raise ValueError("No character detected — image may be blank after processing.")

    ys, xs = np.indices(arr.shape)
    cx = int(np.round((xs * arr).sum() / total))
    cy = int(np.round((ys * arr).sum() / total))

    # Paste so centre of mass lands at (32, 32)
    paste_x = 32 - cx
    paste_y = 32 - cy
    canvas.paste(img, (paste_x, paste_y))

    # ---- Normalise to [-1, 1] ----
    tensor = np.array(canvas, dtype=np.float32) / 255.0   # [0, 1]
    tensor = tensor * 2.0 - 1.0                           # [-1, 1]
    tensor = torch.tensor(tensor).unsqueeze(0).unsqueeze(0)  # [1,1,64,64]
    return tensor


# ──────────────────────────────────────────────
# 3. LABEL LOADING
# ──────────────────────────────────────────────

def load_labels():
    """
    Tries to load class labels from the data/processed/train folder.
    Falls back to class index numbers if labels are not found.
    """
    label_path = os.path.join("data", "processed", "train")
    if os.path.isdir(label_path):
        labels = sorted(os.listdir(label_path), key=lambda x: int(x) if x.isdigit() else x)
        return labels
    return [str(i) for i in range(156)]


# ──────────────────────────────────────────────
# 4. MAIN PREDICTION
# ──────────────────────────────────────────────

def predict(image_path):
    print(f"\n📷  Image     : {image_path}")

    # Load model
    model_path = "tamil_net.pt"
    if not os.path.exists(model_path):
        print("❌  ERROR: tamil_net.pt not found!")
        print("   Make sure you run this script from the TamilNet project folder.")
        sys.exit(1)

    print("🧠  Loading model ...")
    net = TamilNet(num_classes=156)
    net.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    net.eval()

    # Load labels
    labels = load_labels()

    # Preprocess image
    print("🔧  Preprocessing image ...")
    try:
        tensor = preprocess_image(image_path)
    except Exception as e:
        print(f"❌  Could not process image: {e}")
        sys.exit(1)

    # Predict
    with torch.no_grad():
        output = net(tensor)
        probs  = F.softmax(output, dim=1)
        top5_probs, top5_idx = torch.topk(probs, 5)

    print("\n" + "═" * 40)
    print("  🏆  TOP PREDICTIONS")
    print("═" * 40)
    for rank, (idx, prob) in enumerate(zip(top5_idx[0], top5_probs[0]), 1):
        char  = labels[idx.item()] if idx.item() < len(labels) else f"class_{idx.item()}"
        conf  = prob.item() * 100
        bar   = "█" * int(conf / 5)
        print(f"  {rank}. {char:>8}   {conf:5.1f}%  {bar}")
    print("═" * 40)

    best_char = labels[top5_idx[0][0].item()] if top5_idx[0][0].item() < len(labels) else "Unknown"
    best_conf = top5_probs[0][0].item() * 100
    print(f"\n✅  Predicted: {best_char}  (confidence: {best_conf:.1f}%)\n")


# ──────────────────────────────────────────────
# 5. ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n❗  Usage: python predict_image.py <path_to_image>")
        print("   Example: python predict_image.py my_photo.jpg\n")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"\n❌  File not found: {image_path}\n")
        sys.exit(1)

    predict(image_path)