# 📘 Introduction

This is a lightweight hybrid model of CNN and Transformer, which only have 600k parameters and achieves 93.1% accuracy on Fasion Mnist Dataset. 

---

## 📝 目录

- [Introduction](#Introduction)
- [功能特点](#功能特点)
- [安装方法](#安装方法)
- [使用方法](#使用方法)
- [项目结构](#项目结构)
- [模型训练 / 数据使用（可选）](#模型训练--数据使用可选)
- [贡献方式](#贡献方式)
- [许可证](#许可证)
- [致谢](#致谢)

---

## 📖 Project Introduction

Fashion-MNIST is a dataset of Zalando's article images, consisting of 60,000 training examples and 10,000 test examples. Each example is a 28x28 grayscale image associated with one of 10 fashion categories.

This project builds a hybrid model combining CNN and Transformer architectures. Specifically, the CNN is used to extract local features, followed by a Transformer to capture global features. The model contains only about 600k parameters, making it lightweight and suitable for training on most devices. After 90 epochs of training, it achieves an impressive accuracy of 93.1%.

Finally, Grad-CAM is used to visualize and interpret the model’s decision-making process by highlighting the important regions in the input image that influence predictions.


---

## ✨ 功能特点

- ✅ 数据预处理与加载  
- ✅ 模型训练与验证  
- ✅ Grad-CAM 可视化  
- ✅ 支持 CLI 参数配置  
- ✅ 支持 CPU/GPU  

---

## 🛠️ 安装方法

```bash
git clone https://github.com/James-sjt/FashionMnist.git
cd FashionMnist
