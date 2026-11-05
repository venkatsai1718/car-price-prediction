# Car Price Prediction

A Flask web app that predicts the resale price of a used car based on details like company, model, year, fuel type, and kilometres driven. Built with a Linear Regression model trained on Quikr used car listings.

**Live demo:** [car-price-prediction-s1ty1.vercel.app](https://car-price-prediction-s1ty1.vercel.app)

---

## Features

- Dropdowns for car company, model, year of purchase, and fuel type
- Input for kilometres driven
- Instant price prediction via a single form submission

---

## Project structure

```
car-price-prediction/
├── app.py                  # Flask application and /predict route
├── LR.pkl                  # Trained Linear Regression model
├── quikr_cleaned.csv       # Cleaned Quikr dataset used for training
├── requirements.txt        # Python dependencies
├── static/
│   └── css/                # Stylesheets
└── templates/
    └── index.html          # Main UI template
```

---

## Local development

```bash
git clone https://github.com/venkatsai1718/car-price-prediction.git
cd car-price-prediction
pip install -r requirements.txt
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## How it works

The form submits a POST request to `/predict` with these fields:

| Field | Description |
|---|---|
| `company` | Car manufacturer |
| `car_model` | Specific model |
| `year` | Year of purchase |
| `fuel_type` | Petrol / Diesel / CNG / LPG / Electric |
| `kms_driven` | Kilometres driven |

The app loads `LR.pkl` (a scikit-learn Linear Regression pipeline) and returns the predicted price as JSON.

---

## Tech stack

| Technology | Version |
|---|---|
| Python | 3 |
| Flask | 3.0.3 |
| scikit-learn | 1.7.1 |
| pandas | 2.2.2 |
| numpy | 1.26.4 |
| gunicorn | 23.0.0 |
| Vercel | — |

---

## Dataset

Training data sourced from **Quikr** used car listings (`quikr_cleaned.csv`).
