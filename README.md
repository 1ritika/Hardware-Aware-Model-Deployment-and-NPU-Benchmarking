# ⚙️ Hardware-Aware Model Deployment and NPU Benchmarking

This project focuses on **deploying and benchmarking deep learning models** (CNN and Transformer) across different hardware backends — **CPU**, **AMD GPU**, and **NPU** — using **ONNX Runtime**. The goal is to analyze performance trade-offs in precision formats (FP32, FP16, INT8) for edge-efficient inference.

---

## 🚀 Overview
- Converted PyTorch-trained CIFAR-10 models (CNN and Transformer) into **ONNX format** for cross-device deployment.  
- Evaluated inference latency, throughput, and accuracy across heterogeneous compute devices.  
- Benchmarked model performance under **mixed precision (FP32 / FP16 / INT8)** configurations.  
- Compared **NPU acceleration efficiency** against CPU and GPU baselines.  

---

## 📈 Key Results
| Hardware | Precision | Avg. Inference Speed-up | Notes |
|-----------|------------|--------------------------|--------|
| CPU | FP32 | 1× (baseline) | Consistent accuracy |
| AMD GPU | FP16 | **2.8× faster** | Minor accuracy drop |
| NPU | INT8 | **4.5× faster** | Best throughput, reduced energy cost |

> ONNXRuntime’s hardware-agnostic execution enables efficient deployment across devices with minimal code changes.

---

## 🧠 Technical Details
- **Models:** CNN and Transformer classifiers on CIFAR-10.  
- **Frameworks:** PyTorch → ONNX conversion → ONNX Runtime execution.  
- **Metrics:** Inference latency, speed-up factor, and precision–accuracy trade-off.  
- **Profiling Tools:** ONNX Runtime Profiler and PyTorch CUDA events.

---

## 🛠️ Requirements
```bash
pip install torch torchvision onnx onnxruntime numpy matplotlib
