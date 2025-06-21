
# 🧠 PROJECTx1: Sentiment Analysis Using CNN

| 📘 Field       | 📝 Details                              |
|----------------|------------------------------------------|
| 👨‍🎓 Course     | Artificial Intelligence (SIT2025)        |
| 🏢 Internship  | GENZ EDUCATEWING                        |
| 📅 Date        | 18-06-2025                              |
| 👨‍💻 Author     | SOMAPURAM UDAY                          |

---

## 📌 Objective

Build a deep learning model using **Convolutional Neural Networks (CNN)** to classify movie reviews as either **Positive** or **Negative**.

---

## 📚 Dataset Used

- **Dataset**: [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Source**: Keras Datasets
- **Classes**:
  - `0` → Negative Review  
  - `1` → Positive Review  
- **Size**: 50,000 reviews (25K train + 25K test)

---

## 🧠 Tech Stack

| Tool/Library     | Usage                                 |
|------------------|----------------------------------------|
| Python 3.x       | Core programming language              |
| TensorFlow/Keras | Deep Learning model building           |
| NLTK             | Text preprocessing, stopwords          |
| Pandas           | Data manipulation                      |
| NumPy            | Vector operations                      |
| Matplotlib       | Visualization of loss/accuracy         |
| Scikit-learn     | Evaluation metrics, confusion matrix   |

---

## 🚀 How It Works

### ✅ Environment Setup

Install required libraries (if running locally):

```bash
pip install tensorflow keras nltk pandas matplotlib scikit-learn
```

Or use the **Open in Colab** button below to run online without installing anything:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/udaycodespace/GENZ-EDUCATEWING-AI-PROJECTS/blob/main/SENTIMENT_ANALYSIS_CNN/code/Sentiment_Analysis_Using_CNN.ipynb)

---

### 🧹 Preprocessing Steps

- Convert to lowercase  
- Remove HTML tags  
- Handle negations using regex  
- Remove stopwords using NLTK  
- Tokenize text and pad sequences

---

### 🏗️ CNN Model Architecture

- Multiple **Conv1D layers** with kernel sizes `[3, 4, 5]`  
- **Global MaxPooling** after each convolution  
- **Dropout** layers for regularization  
- Dense + Sigmoid layer for binary classification

---

### 🎯 Training Details

| Setting            | Value                         |
|--------------------|-------------------------------|
| Epochs             | 10 (can be reduced to 3–5)    |
| Optimizer          | Adam                          |
| Loss Function      | Binary Cross-Entropy          |
| Batch Size         | 64                            |
| Validation Split   | 0.2                           |
| Time to Train      | ~2–4 minutes on Colab GPU     |
| EarlyStopping      | Enabled                       |

---

## 📊 Evaluation

- Accuracy and Loss Graphs  
- Confusion Matrix  
- Classification Report  
- AUC-ROC Score

| Metric                       | Description                                                     |
|-----------------------------|-----------------------------------------------------------------|
| **Accuracy**                | Overall correctness of predictions                             |
| **Precision**               | How many predicted positives were correct                      |
| **Recall**                  | How many actual positives were detected                        |
| **F1-Score**                | Balance between precision and recall                           |
| **AUC-ROC**                 | Model’s ability to distinguish classes                         |
| **Binary Cross-Entropy**    | Loss function used for binary classification                   |
| **Confusion Matrix**        | Visual breakdown of TP, TN, FP, FN                             |

---

## 🔄 How to Run

### 🟢 Online (Recommended)
1. Click the [Open in Colab](https://colab.research.google.com/github/udaycodespace/GENZ-EDUCATEWING-AI-PROJECTS/blob/main/SENTIMENT_ANALYSIS_CNN/code/Sentiment_Analysis_Using_CNN.ipynb) button.
2. Runtime > Change Runtime Type > Select **GPU**
3. Click **Run All**
4. Done! Training + Evaluation will complete in **~2–4 minutes**

---

### 🖥️ Offline (Optional)
1. Clone this repo  
2. Open the notebook:  
   ```
   GENZ-EDUCATEWING-AI-PROJECTS/SENTIMENT_ANALYSIS_CNN/code/Sentiment_Analysis_Using_CNN.ipynb
   ```
3. Install dependencies (see setup above)
4. Run all cells

---

## ✍️ Author

**SOMAPURAM UDAY**  
AI Intern – GENZ EDUCATEWING  
📧 Email: [229x1a2856@gprec.ac.in](mailto:229x1a2856@gprec.ac.in)  
🔗 GitHub: [udaycodespace](https://github.com/udaycodespace)

---

## 📌 Notes

- This project is part of the GENZ EDUCATEWING Internship Program.  
- Designed for hands-on learning in NLP with deep learning models.

