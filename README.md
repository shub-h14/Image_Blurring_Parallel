# 🚀 Image Blurring using CUDA & OpenMP

This project demonstrates high-performance image processing using both **CUDA (GPU parallelism)** and **OpenMP (CPU parallelism)**. We implement basic operations like:

- 📷 Grayscale conversion
- 🔄 Averaging blur (Box filter)
- ⚡ Edge detection (Sobel filter)
- 📏 Image resizing + blurring

---

## 📦 Features

| Operation        | Serial Time | OpenMP Time | CUDA Time |
|------------------|-------------|-------------|-----------|
| Grayscale        | 0.89s       | 0.22s       | 0.04s     |
| Averaging Filter | 1.23s       | 0.34s       | 0.05s     |
| Sobel Filter     | 1.56s       | 0.41s       | 0.06s     |

---

## 🧰 Technologies

- CUDA C++
- OpenMP
- PPM image format
- GCC / G++

---

## 🗂️ Folder Structure

