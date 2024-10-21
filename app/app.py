from flask import Flask, render_template, request
from PIL import Image
import io
import pickle

import random as random
import pandas as pd
import numpy as np
from skimpy import skim
from scipy import stats
import pickle

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, RFE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import requests
from PIL import Image
from io import BytesIO 

import ray
import optuna
from multiprocessing import Pool, get_context
from multiprocessing.pool import ThreadPool

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU, Dropout, BatchNormalization

# Initialize Flask app
app = Flask(__name__)

# Feature engineering function for images
def feature_engineer_image(image):
    # Example feature engineering: Convert the image to grayscale and resize it
    processed_image = image.convert('L')  # Convert to grayscale
    resized_image = processed_image.resize((224, 224))  # Example: Resize to 224x224
    return resized_image

def run_PCA(X_train, X_test, use_all=True, num_to_use=3, num_components=3):
    
    if use_all:
        pca = PCA()
        X_train_PCA = pca.fit_transform(X_train)[:,:num_to_use]
        X_test_PCA = pca.transform(X_test)[:,:num_to_use]

        return X_train_PCA, X_test_PCA
    
    pca = PCA(num_components)
    return pca.fit_transform(X_train), pca.transform(X_test)

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

# Route for the home page (upload form)
@app.route('/')
def index():
    return render_template('upload.html')

# Route to handle image uploads and process them
# Route to handle image uploads and process them
@app.route('/upload', methods=['POST'])
def upload():
    # List of image file inputs from the form
    file_keys = ['photo1', 'photo2', 'photo3', 'photo4', 'photo5']
    
    # Check if all 5 images are uploaded
    files = []
    for key in file_keys:
        if key not in request.files or request.files[key].filename == '':
            return "Please upload all 5 images."
        files.append(request.files[key])

    # Convert the uploaded images to PIL Image objects and store in a list
    img_list = []
    for file in files:
        img = Image.open(file.stream)  # Open the uploaded image directly from memory
        img = img.resize((224, 224))   # Resize image to 224x224 (ResNet50 input size)
        img_list.append(img)

    # Now pass the list of images to the feature extraction function
    features = prepare_extract_features(img_list)

    # For demo purposes, we can return the length of the extracted features
    return f"Successfully extracted {len(features)} features!"

if __name__ == '__main__':
    app.run(debug=True)
