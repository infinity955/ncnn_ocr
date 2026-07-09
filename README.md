# ncnn_ocr — 多模型 OCR 纯 C++ 推理

> HunYuanOCR / GLM-OCR / PaddleOCR-VL-1.6 — 三个模型，一个 `ncnn_ocr.exe`，纯 C++ + ncnn，0 自定义层。

## 快速开始

```bash
cmake -S src -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release -j

# HunYuanOCR
build/Release/ncnn_ocr.exe --model . --image assets/testimg.jpg

# GLM-OCR
build/Release/ncnn_ocr.exe --model assets/glm_ocr --image assets/testimg.jpg --prompt "识别图片中的文字"

# PaddleOCR-VL-1.6
build/Release/ncnn_ocr.exe --model assets/paddleocr_vl --image assets/testimg.jpg --prompt ""
```

## 文档

| 文档 | 内容 |
|------|------|
| [tutorial/README_CN.md](tutorial/README_CN.md) | 📘 **完整使用教程**（环境搭建 / 模型导出 / 编译 / 运行） |
| [showcase/](showcase/) | 🎯 成果展示（三个模型的转换难点与结果对比） |
