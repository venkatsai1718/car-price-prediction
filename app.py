from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

file_path = os.path.join(os.path.dirname(__file__), 'quikr_cleaned.csv')
model_path = os.path.join(os.path.dirname(__file__), 'LR.pkl')
car = pd.read_csv(file_path)

@app.route("/")
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()
    return render_template('index.html', companies=companies, car_models= car_models, year=year, fuel_types=fuel_type)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form['company']
    model = request.form['model']
    y = request.form['year']
    fuel = request.form['fuel_type']
    kms_driven = request.form['kms_driven']
            
    pipe = pickle.load(open(model_path, 'rb'))
    input_data = pd.DataFrame([{
            'company': company,
            'name': model,
            'year': int(y),
            'kms_driven': int(kms_driven),
            'fuel_type': fuel
        }])
    prediction = np.round(pipe.predict(input_data)[0])
    print(company, model, y, fuel, kms_driven, prediction)
    return jsonify({"result": str(prediction)})

if __name__=='__main__':
    app.run(debug=True)