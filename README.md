
# 🎨 Neural Style Transfer with PyTorch

This project implements a Neural Style Transfer system using a pre-trained **VGG-19** network to merge the content of one image with the artistic style of another.

---

## ✅ Deliverable
> A Python script that generates a stylized image by combining a content image and a style image using a CNN-based algorithm.

---


---

## 🛠 Requirements

Install the required Python packages:

```bash
pip install torch torchvision pillow
 
🧠 How It Works
Content Image: The base structure to retain.

Style Image: The texture and color scheme to apply.

Model: Pre-trained VGG-19 extracts deep features from both images.

Optimization: Iteratively adjusts a copy of the content image to match the style.

📌 Notes
Ensure the images/ directory contains valid .jpg or .png files.

Image sizes are automatically scaled (max 400px on longest side).

GPU acceleration is supported via CUDA (if available).

