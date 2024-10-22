from flask import Flask, render_template, request
from PIL import Image
import io
import pickle
from scipy import stats
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)

def prepare_extract_features(img_arr):
    features = []
    resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    for img in img_arr:
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features_img = resnet.predict(img_array)
        features.extend(features_img[0])
    return features

def run_model(features):
    predictions = []
    for j in range(5):
        model_path = f'../app/models/random_forest_model_{j}.pkl'
        file_path = f'../app//models/pca_model{j}.pkl'
        with open(file_path, 'rb') as file:
            pca = pickle.load(file)
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        feature_vector = pca.transform(np.array(features[2048*j:2048 + 2048*j]).reshape(1,-1))
        prediction = model.predict(np.array(feature_vector).reshape(1,-1))
        predictions.append(prediction)
    
    return stats.mode(predictions)[0][0].squeeze()

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    file_keys = ['photo1', 'photo2', 'photo3', 'photo4', 'photo5']
    files = []
    for key in file_keys:
        if key not in request.files or request.files[key].filename == '':
            return "Please upload all 5 images."
        files.append(request.files[key])

    img_list = []
    for file in files:
        img = Image.open(file.stream)
        img = img.resize((224, 224))
        img_list.append(img)

    features = prepare_extract_features(img_list)
    prediction = run_model(features)
    result = "This restaurant is considered expensive." if prediction == 1 else "This restaurant is considered cheap."
    return render_template('upload.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
