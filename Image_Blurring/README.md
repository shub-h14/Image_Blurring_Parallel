# 🚀 Image Blurring using CUDA & OpenMP

This project implements high-performance parallel image processing using OpenMP for CPU and CUDA for GPU acceleration. It supports:

- 📷 Grayscale Conversion
- 🔄 Averaging Filter (Box Blur)
- ⚡ Sobel Edge Detection (extendable)
- 📏 Image Resizing + Blurring

---

## 📦 Features

| Operation        | Serial Time | OpenMP Time | CUDA Time |
|------------------|-------------|-------------|-----------|
| Grayscale        | 0.89s       | 0.22s       | 0.04s     |
| Averaging Filter | 1.23s       | 0.34s       | 0.05s     |
| Sobel Filter     | 1.56s       | 0.41s       | 0.06s     |

---

## 🛠 Technologies

- CUDA (GPU programming)
- OpenMP (CPU parallelism)
- C++ / C
- PPM image format

---

## 🧰 Folder Structure

