from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import traceback
import base64
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')

# Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load model
MODEL_PATH = os.getenv('MODEL_PATH', 'animal_model.keras')
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

class_names = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant', 'Giraffe',
               'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('User already exists. Please login.')
            return redirect(url_for('login'))
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.')
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', name=current_user.username)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        if model is None:
            flash("Model not loaded. Cannot make predictions.")
            return redirect(url_for('dashboard'))

        if 'images' not in request.files:
            flash("No files uploaded.")
            return redirect(url_for('dashboard'))

        files = request.files.getlist('images')
        if len(files) == 0 or files[0].filename == '':
            flash("No files selected.")
            return redirect(url_for('dashboard'))

        all_predictions = []
        
        for image_file in files:
            if image_file.filename == '':
                continue
            
            img = Image.open(image_file).convert('RGB')
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            preds = model.predict(img_array)[0]
            top_indices = preds.argsort()[-3:][::-1]
            top_preds = [(class_names[i], float(preds[i])) for i in top_indices]
            
            # Thumbnail
            buffered = BytesIO()
            img.thumbnail((150, 150))
            img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            all_predictions.append({
                'filename': image_file.filename,
                'predictions': top_preds,
                'image_base64': img_base64
            })

        return render_template('dashboard.html', name=current_user.username, batch_predictions=all_predictions)

    except Exception as e:
        print(traceback.format_exc())
        flash("Error during prediction. Please try again.")
        return redirect(url_for('dashboard'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.')
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
