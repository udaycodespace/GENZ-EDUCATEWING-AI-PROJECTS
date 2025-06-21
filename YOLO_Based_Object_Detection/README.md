# 🧠 PROJECTx2: Object Detection Using YOLOv8

## 📘 Field	📝 Details  
- **👨‍🎓 Course**: Artificial Intelligence (SIT2025)  
- **🏢 Internship**: GENZ EDUCATEWING  
- **📅 Date**: 21-06-2025  
- **👨‍💻 Author**: SOMAPURAM UDAY  

## 📌 Objective  
Build an object detection system using YOLOv8 that can detect multiple objects in real-time from images, uploaded files, or webcam input.

## 📚 Dataset Used  
- **Dataset**: COCO128 (optional training)  
- **Source**: Ultralytics  
- **Classes**: 80 Common Objects (COCO categories)  
- **Size**: Small subset (128 images)  

## 🧠 Tech Stack  
| Tool/Library | Usage |
|--------------|-------|
| Python 3.x | Core programming language |
| Ultralytics | YOLOv8 models, training, inference |
| OpenCV | Webcam input, image processing |
| Matplotlib | Visualizing results |
| Google Colab | Cloud-based execution |
| JavaScript | Webcam capture in Colab |

## 🚀 How It Works  

### ✅ Environment Setup  
Install dependencies:
```
pip install ultralytics opencv-python matplotlib
```
Or use Colab for hassle-free setup.  
**Note:** Colab webcam integration supports only **short video capture** using JS hacks.

### 🧹 Preprocessing Steps  
- Create project folder  
- Download sample images (e.g., zidane.jpg, bus.jpg)  
- Prepare custom upload or capture image from webcam  

### 🏗️ YOLOv8 Inference Flow  
- Load pretrained model (`yolov8n.pt`)  
- Use `model.predict()` on images  
- Save and visualize results  

## 🎯 Training Details (Optional)  
| Setting | Value |
|---------|-------|
| Model | YOLOv8 Nano |
| Dataset | coco128.yaml |
| Epochs | 3 |
| Image Size | 640 |

Command:  
```python
model.train(data="coco128.yaml", epochs=3, imgsz=640)
```

## 📊 Evaluation  
| Metric | Description |
|--------|-------------|
| mAP@0.5 | Mean average precision at IoU 0.5 |
| Precision | Correctness of positive predictions |
| Recall | Coverage of true positives |
| Confusion Matrix | Available via metrics |

## 🔄 How to Run  

### 🟢 Online (Recommended via Colab)  
1. Open notebook in Google Colab  
2. Runtime > Change runtime type > GPU  
3. Run All  
4. Upload your own image or use webcam to test  
⚠️ Webcam capture supports still frame only, not continuous video.

### 🖥️ Offline (Optional)  
1. Clone the repo  
2. Install dependencies  
3. Open Python notebook/script  
4. Run inference on test images  

## ✍️ Author  
**SOMAPURAM UDAY**  
AI Intern – GENZ EDUCATEWING  
📧 Email: 229x1a2856@gprec.ac.in  
🔗 GitHub: [udaycodespace](https://github.com/udaycodespace)

## 📌 Notes  
- Designed for demo and learning purposes  
- Future work can include detection on full video streams, Streamlit UI, or real-time IoT applications.
