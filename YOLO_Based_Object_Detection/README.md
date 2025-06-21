# 🎯 PROJECTx2: Object Detection Using YOLOv8

## 📘 Field  
| Category       | Details                        |
|----------------|--------------------------------|
| 👨‍🎓 **Course**     | Artificial Intelligence (SIT2025) |
| 🏢 **Internship** | GENZ EDUCATEWING               |
| 📅 **Date**       | 21-06-2025                     |
| 👨‍💻 **Author**     | SOMAPURAM UDAY                 |

---

> ⚠️ **Note:** This project is designed to run on **Google Colab**.  
> - Webcam capture and file upload use **Colab-only JavaScript features**.  
> - Real-time object detection **works best on images and very short videos** only.  
> - Avoid using large videos — YOLOv8 inference in Colab has runtime/memory limits.

---

### 🔧 Steps to Execute (Quick Guide):

1. **Open in Google Colab**
2. **Install dependencies** (`!pip install -q ultralytics opencv-python`)
3. **Download sample image / upload your own**
4. **Run YOLOv8 inference**
5. **(Optional)** Train on small dataset (`coco128.yaml`)
6. **Validate model performance**
7. **Use webcam (Colab only)** for live detection

---

## 📌 Objective

Build an end-to-end Object Detection System using **YOLOv8** (You Only Look Once) that can:

- 🎯 Detect multiple objects in real-time images or webcam input  
- 📷 Perform image inference on uploaded files  
- 📊 Optionally train YOLOv8 on a mini dataset (`coco128.yaml`)  
- ✅ Evaluate model performance using `val()` method  

## 🧪 Step-wise Implementation

---

### ✅ Step 1: Install Dependencies & Setup 

```python
!pip install -q ultralytics
import os
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
```

### ✅ Step 2: Download Sample Image & Load YOLOv8

```python
# Create directories
os.makedirs("yolo_project/images", exist_ok=True)

# Download a sample image
!wget -q https://ultralytics.com/images/zidane.jpg -O yolo_project/images/zidane.jpg

# Ultralytics has a built-in dataset 'coco128.yaml' for quick training
```

### ✅ Step 3: Optional - Quick Training on COCO128 Dataset (⚠️ Takes Time!)

```python
model = YOLO('yolov8n.pt')  # YOLOv8 nano
model.train(data='coco128.yaml', epochs=3, imgsz=640)  # quick train
```

### ✅ Step 4: Run Inference on Image

```python
# Image Inference
results = model("yolo_project/images/zidane.jpg")
results[0].save(filename="yolo_project/images/result.jpg")

# Download another test image
!wget -q https://ultralytics.com/assets/bus.jpg -O yolo_project/images/bus.jpg
results = model("yolo_project/images/bus.jpg")
results[0].save(filename="yolo_project/images/bus_result.jpg")

# Display the result
img = Image.open("yolo_project/images/result.jpg")
plt.imshow(img)
plt.axis('off')
plt.title("YOLOv8 Detection Output (Image)")
plt.show()
```

### ✅ Step 5: Model Evaluation (Validation)

```python
metrics = model.val()
print(metrics)  # Shows mAP, precision, recall
```

### ✅ Step 6: Test on Another Image

```python
from google.colab import files
uploaded = files.upload()  # 📤 Select image from your computer

import shutil
uploaded_filename = list(uploaded.keys())[0]
dest_path = f"yolo_project/images/{uploaded_filename}"
shutil.move(uploaded_filename, dest_path)

# Run YOLOv8 detection
results = model(dest_path)
results[0].save(filename="yolo_project/images/result_custom.png")

# Show result image
img = Image.open("yolo_project/images/result_custom.png")
plt.imshow(img)
plt.axis('off')
plt.title("Detected Objects - Your Uploaded Image")
plt.show()
```

### ✅ Step 7: Real-Time Webcam Detection (Colab JS Integration)

```python
!pip install -q opencv-python

import base64
from IPython.display import display, Javascript
from google.colab.output import eval_js
from google.colab.patches import cv2_imshow
import cv2
import numpy as np
import PIL.Image
import io

def capture_image(filename='captured_image.jpg'):
    display(Javascript('''
        async function capture() {
          const div = document.createElement('div');
          const capture = document.createElement('button');
          capture.textContent = '📷 Capture';
          div.appendChild(capture);
          document.body.appendChild(div);

          const video = document.createElement('video');
          video.style.display = 'block';
          const stream = await navigator.mediaDevices.getUserMedia({video: true});
          document.body.appendChild(video);
          video.srcObject = stream;
          await video.play();

          google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

          await new Promise((resolve) => capture.onclick = resolve);

          const canvas = document.createElement('canvas');
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          canvas.getContext('2d').drawImage(video, 0, 0);
          stream.getVideoTracks()[0].stop();
          video.remove();
          capture.remove();
          div.remove();

          const dataURL = canvas.toDataURL('image/jpeg');
          return dataURL;
        }
        capture();
    '''))

    data = eval_js("capture()")
    binary = io.BytesIO(base64.b64decode(data.split(',')[1]))
    img = PIL.Image.open(binary)

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img_cv)
    print(f"✅ Image saved as {filename}")
    return filename

captured_file = capture_image('webcam_input.jpg')

results = model("webcam_input.jpg")
results[0].save(filename="detection_result.jpg")

img = Image.open("detection_result.jpg")
plt.imshow(img)
plt.axis("off")
plt.title("YOLOv8 Detection Result")
plt.show()
```

## 🧠 Concepts Used

- **YOLOv8 (You Only Look Once):** Real-time object detector  
- **Model Training:** Used `coco128.yaml` for fast training  
- **Inference:** Direct detection on images, uploads, webcam  
- **Evaluation:** `mAP@0.5`, Precision, Recall via `model.val()`  
- **Visualization:** Matplotlib + PIL to show detection output  
- **Webcam Integration:** Captured real-time frame using JS in Colab  

---

## ⚙️ Tools & Libraries

| Tool          | Purpose                          |
|---------------|----------------------------------|
| **Ultralytics** | YOLOv8 models & training         |
| **OpenCV**      | Image capture & frame processing |
| **Matplotlib**  | Result visualization             |
| **Google Colab**| Web-based Python environment     |
| **JavaScript**  | Capture image from webcam in Colab|

---

## ⚠️ Challenges

| Issue                   | Solution                           |
|-------------------------|------------------------------------|
| 🔴 Video file not found | Switched to working sample images  |
| ⚠️ Webcam lag in Colab  | Used JS + `eval_js()` workaround   |
| 📂 File not saved       | Used `results[0].save()` method    |

---

## 🔭 Future Enhancements

- Allow video file uploads for detection  
- Integrate with **Streamlit** or **Gradio** for a full app  
- Experiment with **YOLOv8s**, **YOLOv8m**, or **YOLOv8x** for better accuracy  
- Use custom training data for real use-cases (e.g., helmet detection, face masks)

---

## ✅ Summary

A complete real-time object detection system was implemented using **YOLOv8**.  
The project includes **image inference**, **webcam-based detection**, and **optional training** on the `coco128` dataset.  
Detection results were **visualized**, and model performance was **validated** with standard evaluation metrics.  
The setup is ideal for **rapid prototyping** and **learning object detection workflows**.

