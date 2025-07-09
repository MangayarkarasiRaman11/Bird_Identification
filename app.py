from flask import Flask, request, jsonify, render_template, url_for
from PIL import Image
import io
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load the ResNet50 model
model = ResNet50(weights='imagenet')

def predict_bird_species(img):
    img = img.resize((224, 224))  # Resize for ResNet50
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    return {label.replace('_', ' ').title(): f"{prob * 100:.2f}%" for (_, label, prob) in decoded_predictions}

# Route for Home Page
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

# Route for Bird Identification Page
@app.route('/identify', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        img = Image.open(io.BytesIO(file.read()))
        predictions = predict_bird_species(img)

        return jsonify({'predictions': predictions})

    return render_template('upload.html')

# Route for About Page
@app.route('/about')
def about():
    return render_template('about.html')

# Route for Help Page
@app.route('/help')
def help_page():
    return render_template('help.html')

if __name__ == '__main__':
    app.run(debug=True)
