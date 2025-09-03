import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Config
UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables to hold model and class names
model = None
class_names = []

# Create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_data():
    """Load images and labels from data folder"""
    images = []
    labels = []
    global class_names
    class_names = sorted(os.listdir(UPLOAD_FOLDER))
    for idx, class_name in enumerate(class_names):
        class_folder = os.path.join(UPLOAD_FOLDER, class_name)
        if not os.path.isdir(class_folder):
            continue
        for fname in os.listdir(class_folder):
            if allowed_file(fname):
                fpath = os.path.join(class_folder, fname)
                img = Image.open(fpath).convert('RGB').resize((224, 224))
                img = img_to_array(img)
                img = preprocess_input(img)
                images.append(img)
                labels.append(idx)
    if len(images) == 0:
        return None, None
    return np.array(images), np.array(labels)

def build_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base_model.trainable = False  # Freeze base model

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

@app.route('/', methods=['GET', 'POST'])
def index():
    global model, class_names
    if request.method == 'POST':
        # Handle class names input and image uploads
        if 'class_names' in request.form:
            # User submitted class names
            raw_classes = request.form['class_names']
            # Split by comma and strip spaces
            classes = [c.strip() for c in raw_classes.split(',') if c.strip()]
            if not classes:
                flash('Please enter at least one class name.')
                return redirect(url_for('index'))
            # Create folders for each class
            for c in classes:
                os.makedirs(os.path.join(UPLOAD_FOLDER, c), exist_ok=True)
            flash(f'Classes created: {", ".join(classes)}. Now upload images for each class.')
            return render_template('upload.html', classes=classes)

        # Handle image uploads per class
        if 'upload_images' in request.form:
            classes = request.form.getlist('classes')
            for c in classes:
                files = request.files.getlist(f'files_{c}')
                class_folder = os.path.join(UPLOAD_FOLDER, c)
                for file in files:
                    if file and allowed_file(file.filename):
                        filename = secure_filename(file.filename)
                        file.save(os.path.join(class_folder, filename))
            flash('Images uploaded successfully.')
            return redirect(url_for('index'))

        # Handle training
        if 'train' in request.form:
            images, labels = load_data()
            if images is None:
                flash('No images found for training. Please upload images first.')
                return redirect(url_for('index'))
            model = build_model(len(class_names))
            model.fit(images, labels, epochs=5, batch_size=8)
            flash('Training completed.')
            return redirect(url_for('index'))

        # Handle classification
        if 'classify' in request.form:
            if model is None:
                flash('Model not trained yet. Please train the model first.')
                return redirect(url_for('index'))
            file = request.files.get('test_image')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join('temp', filename)
                os.makedirs('temp', exist_ok=True)
                file.save(filepath)
                img = Image.open(filepath).convert('RGB').resize((224,224))
                x = img_to_array(img)
                x = preprocess_input(x)
                x = np.expand_dims(x, axis=0)
                preds = model.predict(x)
                pred_class = class_names[np.argmax(preds)]
                os.remove(filepath)
                return render_template('result.html', prediction=pred_class)
            else:
                flash('Please upload a valid image to classify.')
                return redirect(url_for('index'))

    # GET request
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)