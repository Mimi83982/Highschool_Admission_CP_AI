# 🎓 AI Admission & Academic Prediction Portal

> A Flask-based web application that leverages **Machine Learning** to automate student admission evaluation and predict future academic performance.

---

## 📖 Overview

**AI Admission & Academic Prediction Portal** is a web application developed using **Flask** that integrates Machine Learning models to provide intelligent academic decision support.

### ✨ Features

- 🎯 Student Admission Prediction using **Multi-Layer Perceptron (MLP)**
- 📈 Academic Grade Forecasting
- 🖥️ User-friendly Flask Web Interface
- ⚡ Fast prediction with pre-trained Machine Learning models

---

# 🚀 Getting Started

## 1️⃣ Prerequisites

Before running the application, make sure you have installed:

- **Python 3.10.x**

Check your Python version:

```bash
python --version
```

---

## 2️⃣ Create & Activate Virtual Environment

Create a virtual environment (if you haven't already):

```bash
python -m venv venv
```

### Windows

```bash
.\venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

If the `requirements.txt` file does not exist, generate it using:

```bash
pip freeze > requirements.txt
```

---

## 4️⃣ Run the Application

Start the Flask server:

```bash
python generate_data.py
python train_model.py
python train_forecaster.py
python app.py
```

The application will be available at:

```
http://127.0.0.1:5000
```

---

# 🤖 Machine Learning Models

Ensure the following trained model files are located in the same directory as `app.py`.

```
admission_mlp_model.pkl
admission_scaler.pkl
grade_forecaster.pkl
forecaster_scaler.pkl
```

Without these files, the prediction features will not function properly.

---

# 📂 Project Structure

```text
AI-Admission-Portal/
│
├── app.py
├── requirements.txt
├── admission_mlp_model.pkl
├── admission_scaler.pkl
├── grade_forecaster.pkl
├── forecaster_scaler.pkl
│
├── static/
│
├── templates/
│
└── README.md
```

---

# 🛠️ Technology Stack

- Python 3.10
- Flask
- Scikit-learn
- Pandas
- NumPy
- Joblib
- HTML5
- CSS3
- JavaScript

---

# 📌 Notes

- Always activate the virtual environment before running the project.
- Ensure all Machine Learning model files are present.
- Install dependencies before starting the application.
- If you encounter missing package errors, reinstall dependencies using:

```bash
pip install -r requirements.txt
```

---

# 👨‍💻 Authors

Developed as an AI-based academic decision support system using **Flask** and **Machine Learning**.

---

## 📄 License

This project is intended for educational and research purposes.
