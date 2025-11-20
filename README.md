# 🌟 GenoPredict – DNA Sequence Disease Prediction System  
*A full-stack machine learning web application for predicting diseases from DNA sequence patterns.*

---

## 🧬 Overview

**GenoPredict** is an end-to-end machine learning application built using **Flask**, **Python**, and **Scikit-learn**.  
It analyzes DNA sequence patterns and predicts the most likely associated disease using a trained ML model.

This project is designed for:

- Bioinformatics + ML students  
- Researchers exploring DNA-based classification  
- Faculty teaching end-to-end ML pipelines  
- College projects requiring UI + backend + ML + authentication  

The system includes:

- 🔬 DNA preprocessing  
- 🤖 ML model (Random Forest)  
- 🌐 Flask dashboard  
- 🔐 Login/Register with hashed passwords  
- 📈 Live analytics & prediction results  
- 🧪 Automated testing scripts

---

## 🚀 Features

### 🧠 1. Machine Learning Model
- Model: **Random Forest Classifier**
- Trained on **15 DNA sequence segments**
- Encodes all sequences using LabelEncoder
- Predicts **10 disease classes**
- Real-time inference in Flask using pickle files

### 📊 2. Modern Interactive Dashboard
- Fully responsive  
- Glass UI + clean animations  
- Information cards:
  - Total Analyses  
  - Model Confidence  
  - High-Risk Cases  
  - Pending Reviews  
- Prediction result card with recommendations  
- Sidebar navigation & user profile section  

### 🔐 3. Built-in Authentication System
- SQLite database  
- User registration  
- Password hashing using Werkzeug  
- Protected routes  
- Personalized dashboard (username displayed)

### 🧪 4. Test Automation
Included testing scripts:

- `test_app.py` → end-to-end test  
- `bulk_test_from_csv.py` → runs 100+ predictions to check stability  

### 💾 5. Solid Backend Architecture
- Clean Flask routing  
- Jinja2 templating  
- Model + encoders loaded only once  
- Secure input processing  

---

## 🏗 Project Structure

```
DNA Sequence Analysis/
│
├── app.py                     # Flask web app
├── MainFile.py                # ML training + preprocessing script
├── dna_sequence_dataset.csv   # DNA dataset
├── label_encoders.pickle      # Sequence encoders
├── model_rf.pickle            # Trained model
│
├── templates/
│   ├── index.html             # Main dashboard
│   ├── login.html             # Login page
│   └── register.html          # Register page
│
├── users.db                   # SQLite user database
│
├── test_app.py                # Test script for the API
└── bulk_test_from_csv.py      # Generates 100+ predictions for evaluation
```

---

## 🧬 Model & Data Explanation

### 🔡 Input Columns (DNA Features)

Your system uses the following **15 sequence windows**:

```
Seq_Window_1  
Seq_Window_2  
Promoter_Seq  
Exon_1_Seq  
Mutation_Site_Seq  
Enhancer_Seq  
Intron_1_Seq  
CpG_Island_Seq  
Repeat_Seq  
Motif_1  
Motif_2  
DNA_Segment_1  
DNA_Segment_2  
DNA_Segment_3  
DNA_Segment_4
```

All features consist of standard nucleotides: **A, C, G, T**.

### 🎯 Output Labels (Diseases Predicted)

The model predicts one of:

- Alzheimer  
- Asthma  
- Breast Cancer  
- Diabetes  
- Heart Failure  
- Lung Cancer  
- Osteoporosis  
- Parkinson  
- Prostate Cancer  
- Stroke  

---

## 🧠 How the ML Pipeline Works

1. Load dataset  
2. Label-encode each DNA window  
3. Train a Random Forest Classifier  
4. Save:
   - Model (`model_rf.pickle`)
   - Encoders (`label_encoders.pickle`)
5. Flask loads these during runtime  
6. User inputs are encoded the same way  
7. Model predicts the disease  
8. UI displays:
   - Predicted disease  
   - Recommended actions  

---

## 🎨 Beautiful UI Features

The dashboard contains:

✔️ DNA background with glass-style blur  
✔️ Smooth animations  
✔️ Card-based layout  
✔️ Auto-updated stats  
✔️ Result cards with check icons  
✔️ Responsive layout for mobile  
✔️ Sidebar and top header  

Makes the entire tool look **premium and industry-grade**.

---

## 🏁 How to Run Locally

### 1. Install all requirements
```bash
pip install flask numpy pandas scikit-learn matplotlib seaborn requests
```

### 2. Train the ML model
```bash
python MainFile.py
```

Generates:
- `model_rf.pickle`
- `label_encoders.pickle`

### 3. Start the web server
```bash
python app.py
```

Open:

```
http://127.0.0.1:5000
```

---

## 🧪 Testing

### ✔️ Basic Functional Test
```bash
python test_app.py
```

### ✔️ Bulk Prediction Test
```bash
python bulk_test_from_csv.py
```

This sends real DNA rows through the prediction engine and prints summary statistics.

---

## 🛡 Security

- Password hashing using `generate_password_hash()`  
- Cookies secured using **Flask secret key**  
- Parameterized SQL queries  
- No raw password storage  
- Session-based login  

---

## 🎉 Why This Project Is Excellent for Students

This project demonstrates:

- ML preprocessing  
- Encoding techniques  
- Model training  
- Flask API + Dashboard  
- Authentication  
- Front-end design  
- File-based model loading  
- Testing frameworks  
- Real-world ML deployment workflow  

Perfect for:

✔️ Semester projects  
✔️ Final-year ML projects  
✔️ Bioinformatics coursework  
✔️ AI mini-projects  
✔️ Resume portfolio  

---

## ❤️ A Personal Note

This project was crafted with attention, clarity, and aesthetics.  
Your DNA dashboard looks polished, professional, and absolutely presentation-ready.