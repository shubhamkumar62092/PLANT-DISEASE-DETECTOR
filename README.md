# 🌿 Plant Disease Detector

An AI-powered web app that detects plant diseases from leaf images using a VGG16 deep learning model.

🔗 **Live Demo:** [https://plant-disease-detector-shu.onrender.com](https://plant-disease-detector-shu.onrender.com)

---

## 🖼️ How It Works

1. Upload a leaf image (JPG, PNG, WEBP)
2. Click **Analyse Leaf**
3. Get instant disease diagnosis with confidence score

---

## 🧠 Model

- **Architecture:** VGG16 (Transfer Learning)
- **Dataset:** New Plant Diseases Dataset (Kaggle)
- **Classes:** 38 plant disease categories
- **Framework:** TensorFlow / Keras

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | FastAPI (Python) |
| Model | VGG16 + TensorFlow |
| Deployment | Render |

---

## 📁 Project Structure

```
plant-disease-detector/
├── app.py                          # FastAPI backend
├── index.html                      # Frontend UI
├── class_names.json                # 38 disease class labels
├── requirements.txt                # Python dependencies
├── .python-version                 # Python 3.11.9
└── vgg16_plant_disease_model.keras # Trained model
```

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/plant-disease-detector.git
cd plant-disease-detector

# Create virtual environment
python -m venv myvenv
myvenv\Scripts\activate   # Windows
# source myvenv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

---

## 🌱 Supported Plants & Diseases

The model can detect diseases in:
Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato and more — across **38 disease categories**.

---

## 📬 Contact

Made with ❤️ by [Shubham kumar](https://github.com/shubhamkumar62092)
