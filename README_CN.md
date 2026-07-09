# ncnn_ocr — 多模型 OCR 纯 C++ 推理

> HunYuanOCR / GLM-OCR / PaddleOCR-VL-1.6 — 三个模型，一个 `ncnn_ocr.exe`，
> 纯 C++ + ncnn，0 自定义层。

## 结果展示

> **说明**：HunYuanOCR 输出格式为 `文字(x1,y1),(x2,y2)` 带坐标框，GLM 和 PaddleOCR-VL 输出纯文字。
> 小图（000001/000166/000258）分辨率低（如 164×60），vision tokens 极少先输出 EOS 停机（GLM/PaddleOCR 空白），这不是 bug，是模型本身对超小图的处理方式不同。

### testimg.jpg（700×394）

| 图片 | HunYuanOCR | GLM-OCR | PaddleOCR-VL |
|------|-----------|---------|-------------|
| ![](assets/test_images/testimg.jpg) | `顺利上岸(231,392),(764,632)LAMAR(446,850),(559,878)05483(934,835),(997,870)` | `顺利上岸` `LAMAR` `054839` | `顺利上岸` `LAMAR` `05483` |

### 000001.jpg（164×60）

| 图片 | HunYuanOCR | GLM-OCR | PaddleOCR-VL |
|------|-----------|---------|-------------|
| ![](assets/test_images/000001.jpg) | `Friend(46,115),(939,927)` | *(无输出)* | `iFriend` |

### 000166.jpg（132×56）

| 图片 | HunYuanOCR | GLM-OCR | PaddleOCR-VL |
|------|-----------|---------|-------------|
| ![](assets/test_images/000166.jpg) | `TALK(0,0),(999,1000)` | *(无输出)* | `iTalk` |

### 000258.jpg（196×56）

| 图片 | HunYuanOCR | GLM-OCR | PaddleOCR-VL |
|------|-----------|---------|-------------|
| ![](assets/test_images/000258.jpg) | `tpti(0,157),(995,1000)` | *(无输出)* | `tpti`*(重复)* |

### 000965.jpg（公式图）

| 图片 | HunYuanOCR | GLM-OCR | PaddleOCR-VL |
|------|-----------|---------|-------------|
| ![](assets/test_images/000965.jpg) | `<pos>AD=BD` | `$\frac{AD}{DG}=\frac{BD}{AD}$` | `\\(\\frac{AD}{DG}=\\frac{BD}{AD}\\)` |

### 000972.jpg（公式图）

| 图片 | HunYuanOCR | GLM-OCR | PaddleOCR-VL |
|------|-----------|---------|-------------|
| ![](assets/test_images/000972.jpg) | `<pos>` | `j=\frac{3}{2}` | *(无输出)* |

### 000997.jpg（公式图）

| 图片 | HunYuanOCR | GLM-OCR | PaddleOCR-VL |
|------|-----------|---------|-------------|
| ![](assets/test_images/000997.jpg) | `<pos>x>-5/2` | `x>-\frac{5}{2}` | `x>-\\frac{5}{2}` |

## 快速开始

```bash
# 构建（先确保 ../ncnn/build 已编译）
cmake -S src -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release -j

# 运行
build/Release/ncnn_ocr.exe --model . --image assets/test_images/testimg.jpg                           # HunYuanOCR
build/Release/ncnn_ocr.exe --model assets/glm_ocr --image assets/test_images/testimg.jpg --prompt "识别图片中的文字"   # GLM-OCR
build/Release/ncnn_ocr.exe --model assets/paddleocr_vl --image assets/test_images/testimg.jpg --prompt ""            # PaddleOCR-VL
```

## 文档导航

| 文档 | 内容 |
|------|------|
| [tutorial/README_CN.md](tutorial/README_CN.md) | 📘 **完整使用教程**（环境搭建 / 模型导出 / 编译 / 运行） |
| [showcase/](showcase/) | 🎯 成果展示（三个模型的转换难点与结果对比） |
