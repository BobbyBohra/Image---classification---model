🐾 Animal Image Classifier & Assistant

An AI-powered application that classifies animal images and provides intelligent insights about the detected animal using computer vision and machine learning.

🚀 Overview

This project uses deep learning techniques to classify animal images into different categories and acts as an assistant by providing additional information such as species details, habitat, and characteristics.

🎯 Features
🖼️ Upload animal images for classification

🤖 Deep learning model for accurate predictions

📊 Displays predicted animal name with confidence score

🧠 Provides additional insights (habitat, diet, facts)

🌐 User-friendly interface for real-time interaction


🛠️ Tech Stack

Python
TensorFlow / Keras or PyTorch
OpenCV (image processing)
NumPy & Pandas
Streamlit (UI)


🧠 Model Details

Used a CNN (Convolutional Neural Network) for image classification
Trained on animal image dataset (e.g., Kaggle dataset)
Applied preprocessing techniques like resizing, normalization, and augmentation


📂 Project Structure

animal-classifier/
│
├── app.py                # Streamlit app
├── model/
│   ├── model.h5         # Trained model
│
├── data/
│   ├── train/
│   ├── test/
│
├── utils/
│   ├── preprocess.py
│
├── requirements.txt
└── README.md


▶️ How to Run

# Clone repo
git clone https://github.com/your-username/animal-classifier.git

# Go to project folder
cd animal-classifier

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py


📊 Output Example
Input: Image of a tiger
Output:
Prediction: Tiger 🐅
Confidence: 95%
Info: Carnivorous animal found in forests and grasslands
