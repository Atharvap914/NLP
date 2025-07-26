


## 🧠 Natural Language Processing (NLP) —

Natural Language Processing (NLP) is a branch of Artificial Intelligence that helps computers understand, interpret, and generate human language. It combines linguistics with machine learning to process text and speech data.

### 🔑 Key Concepts in NLP:
- **Tokenization:** Splitting text into individual words or sentences.
- **Stopwords Removal:** Removing common words like *"and"*, *"the"*, etc., which do not add much meaning.
- **Stemming & Lemmatization:** Reducing words to their root forms. Example: *“running” → “run”*.
- **Vectorization:** Converting text into numerical format using methods like Bag-of-Words or TF-IDF.
- **Classification:** Using machine learning algorithms (e.g., Logistic Regression, Naive Bayes) to categorize text (e.g., spam/ham, fake/real).

---


## 📁 Project 1: Fake News Detection using TF-IDF + Logistic Regression

### 📌 Objective:
Detect whether a news article is real or fake using Natural Language Processing.

### 🧱 Workflow:
1. **Dataset:** `fake_news_train.csv` with `title`, `author`, `label` columns.
2. **Text Preprocessing:**
   - Fill missing values
   - Combine title & author into `content`
   - Apply stemming and stopword removal
3. **Feature Extraction:**
   - TF-IDF Vectorization to convert text into numerical features.
4. **Modeling:** Logistic Regression classifier.
5. **Evaluation:** Confusion matrix and classification report.

### 🛠 Preprocessing Steps:
```
re.sub('[^a-zA-Z]', ' ', content)
content.lower().split()
[stemmer.stem(word) for word in content if word not in stopwords]
```

### ✅ Results:
- Binary classification: `0 = Real`, `1 = Fake`
- Output prints prediction for a new test sample.

---

## 🚀 How to Run

1. Upload the dataset files in your Colab or local environment.
2. Run all the cells in sequence.
3. Check sample predictions and accuracy metrics.

---

## 🧾 Requirements
```
tensorflow
scikit-learn
nltk
pandas
numpy
```

Install them using:
```
pip install tensorflow scikit-learn nltk pandas numpy
```

-## 📁 Project 2: Human Activity Recognition using LSTM

### 📌 Objective:
Predict human physical activities like walking, standing, laying, etc., using accelerometer and gyroscope sensor data from the [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones).

### 🧱 Workflow:
1. **Data Collection:** Sensor data from smartphone (accelerometer & gyroscope).
2. **Preprocessing:** Reshaping and normalizing data.
3. **Modeling:** Sequential model using LSTM (Long Short-Term Memory).
4. **Evaluation:** Classification accuracy and activity-wise predictions.

### 📦 Dataset:
- Shape: `(X_train: [7352, 561])`
- Classes:
  - Walking
  - Walking Upstairs
  - Walking Downstairs
  - Sitting
  - Standing
  - Laying

### 🛠 Model Architecture:
```
Sequential()
├── LSTM(64 units)
├── Dropout(0.5)
├── Dense(64, relu)
└── Dense(6, softmax)
```

### ✅ Results:
- Multi-class classification on 6 physical activities.
- Output shows actual vs predicted labels.
- Real-time prediction simulation using test samples.

