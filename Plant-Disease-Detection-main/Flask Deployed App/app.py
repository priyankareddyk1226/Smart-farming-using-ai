import os
from flask import Flask, redirect, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
import joblib


disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

crop_model = joblib.load('crop_model.pkl')
le_region = joblib.load('le_region.pkl')
le_season = joblib.load('le_season.pkl')
le_crop = joblib.load('le_crop.pkl')

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files.get('image')
        if image is None or image.filename == '':
            return redirect('/index')

        filename = secure_filename(image.filename)
        if filename == '':
            return redirect('/index')

        upload_folder = os.path.join('static', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, filename)
        image.save(file_path)
        print(file_path)

        image = Image.open(file_path)
        image = image.resize((224, 224))
        input_data = TF.to_tensor(image).view((-1, 3, 224, 224))
        output = model(input_data).detach().numpy().flatten()
        exp_output = np.exp(output - np.max(output))
        probabilities = exp_output / exp_output.sum()
        pred = int(np.argmax(probabilities))

        top_indices = np.argsort(probabilities)[::-1][:5]
        top_predictions = [
            {
                'rank': idx + 1,
                'disease_name': disease_info['disease_name'][int(index)],
                'confidence': float(probabilities[int(index)])
            }
            for idx, index in enumerate(top_indices)
        ]

        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

        analysis_metrics = {
            'filename': filename,
            'predicted_index': pred,
            'predicted_confidence': float(probabilities[pred]),
            'top_predictions': top_predictions
        }

        return render_template(
            'submit.html',
            title=title,
            desc=description,
            prevent=prevent,
            image_url=image_url,
            pred=pred,
            sname=supplement_name,
            simage=supplement_image_url,
            buy_link=supplement_buy_link,
            metrics=analysis_metrics,
            disease_row=disease_info.loc[pred].to_dict(),
            supplement_row=supplement_info.loc[pred].to_dict()
        )

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

@app.route('/crop-suggestion', methods=['GET', 'POST'])
def crop_suggestion():
    if request.method == 'POST':
        region = request.form.get('region')
        season = request.form.get('season')
        if not region or not season:
            return render_template('crop_suggestion.html', error="Please select both region and season.")
        region_encoded = le_region.transform([region])[0]
        season_encoded = le_season.transform([season])[0]
        prediction_encoded = crop_model.predict([[region_encoded, season_encoded]])[0]
        predicted_crop = le_crop.inverse_transform([prediction_encoded])[0]
        return render_template('crop_suggestion.html', predicted_crop=predicted_crop, region=region, season=season)
    return render_template('crop_suggestion.html')

if __name__ == '__main__':
    app.run(debug=True)
