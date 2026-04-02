# 🐾 Animal Image Classifier + Voice Assistant

A complete machine learning web application that classifies animal images using a deep learning model (MobileNetV2) and includes a voice assistant powered by **Groq**. The web app supports batch prediction (multiple images at once) with a modern glassmorphic UI.

## ✨ Features

- **Animal Classification**: Upload one or multiple animal images and get top-3 predictions with confidence scores.
- **Batch Prediction**: Process several images in a single request; results include image thumbnails.
- **User Authentication**: Register/login system with password hashing (Flask-Login).
- **Premium UI**: Glassmorphism design, drag & drop, image preview, responsive layout.
- **Voice Assistant** (separate script): Listen to voice commands, answer questions using Groq LLM, open websites, tell time, weather, play music.
- **Trainable Model**: Transfer learning with MobileNetV2 – easy to retrain on custom animal dataset.

## 🧠 Model Details

- Architecture: **MobileNetV2** (pre-trained on ImageNet) + GlobalAveragePooling2D + Dense(128) + Dropout(0.5) + Softmax.
- Input size: 224×224×3.
- Training script: `train_model.py` (supports data augmentation and validation split).
- Model saved as `animal_model.keras`.

## 📁 Project Structure
.
├── .env # Environment variables (API keys, paths)

├── requirements.txt # Python dependencies

├── train_model.py # Train the animal classifier

├── app.py # Flask web application

├── voice_assistant.py # Groq‑based voice assistant

├── animal_model.keras # Trained model (generated after training)

├── users.db # SQLite user database (auto‑created)

├── templates/

│ ├── login.html

│ ├── register.html

│ └── dashboard.html

└── data/

└── train/ # Training images – one subfolder per animal

├── Cat/

├── Dog/

└── ...

▶️ How to Run

# Clone repo

git clone https://github.com/BobbyBohra/Image---classification---model.git

# Go to project folder

cd animal-classifier

# Install dependencies

pip install -r requirements.txt

# Train the Model

python train_model.py

# Run the Web Application

python app.py

# Run the Voice Assistant

python voice.py


🌟 Future Improvements
Real‑time webcam prediction

Export prediction results to CSV

User prediction history

Deploy on cloud (Render / Hugging Face Spaces)

Grad‑CAM heatmaps for model explainability

👤 Author

Made with ❤️ Bobby 
