# EV Adoption Forecasting

As electric vehicle (EV) adoption surges, urban planners need to anticipate infrastructure needs—especially charging stations. Inadequate planning can lead to bottlenecks, impacting user satisfaction and adoption rates. This project addresses that challenge by forecasting EV adoption demand.

## Problem Statement
Using the electric vehicle dataset (which includes information on EV populations, vehicle types, and historical charging usage), create a model to forecast future EV adoption demand based on historical trends in EV growth, types of vehicles, and regional data.

## Goal
Build a regression model that forecasts future EV adoption demand based on historical trends in EV growth, types of vehicles, and regional data.

## Dataset

This dataset shows the number of vehicles that were registered by Washington State Department of Licensing (DOL) each month. The data is separated by county for passenger vehicles and trucks.

### Features:
- **Date**: Counts of registered vehicles are taken on this day (the end of this month). Range: 2017-01-31 to 2024-02-29
- **County**: Geographic region of a state where a vehicle's owner is registered (Vehicles registered in Washington)
- **State**: Geographic region of the country associated with the record
- **Vehicle Primary Use**: Primary intended use of the vehicle (Passenger: 83%, Truck: 17%)
- **Battery Electric Vehicles (BEVs)**: Count of vehicles propelled solely by onboard electric battery
- **Plug-In Hybrid Electric Vehicles (PHEVs)**: Count of vehicles propelled partially from onboard electric battery
- **Electric Vehicle (EV) Total**: Sum of BEVs and PHEVs
- **Non-Electric Vehicle Total**: Count of non-electric vehicles
- **Total Vehicles**: All powered vehicles registered in the county
- **Percent Electric Vehicles**: Comparison of electric vehicles vs non-electric counterparts

**Dataset Link**: [Kaggle - Electric Vehicle Population Size 2024](https://www.kaggle.com/datasets/sahirmaharajj/electric-vehicle-population-size-2024/data)

---

## 📊 Model Performance Metrics

### Model Architecture
- **Type**: Gradient Boosting Regression Model
- **Algorithm**: XGBoost / LightGBM (Time Series Forecasting)
- **Training Data**: 80% of preprocessed EV dataset
- **Testing Data**: 20% of preprocessed EV dataset

### Accuracy Metrics

#### Training Set Performance
| Metric | Value | Description |
|--------|-------|-------------|
| **R² Score** | [Run model_evaluation.py to see] | Proportion of variance explained (0-1, higher is better) |
| **RMSE** | [Run model_evaluation.py to see] | Root Mean Squared Error in vehicle counts |
| **MAE** | [Run model_evaluation.py to see] | Mean Absolute Error - average prediction error |
| **MAPE** | [Run model_evaluation.py to see] | Mean Absolute Percentage Error (%) |
| **Median AE** | [Run model_evaluation.py to see] | Median Absolute Error (robust to outliers) |

#### Testing Set Performance
| Metric | Value | Description |
|--------|-------|-------------|
| **R² Score** | [Run model_evaluation.py to see] | Proportion of variance explained on unseen data |
| **RMSE** | [Run model_evaluation.py to see] | Root Mean Squared Error on test set |
| **MAE** | [Run model_evaluation.py to see] | Mean Absolute Error on test set |
| **MAPE** | [Run model_evaluation.py to see] | Mean Absolute Percentage Error on test set (%) |
| **Median AE** | [Run model_evaluation.py to see] | Median Absolute Error on test set |

### How to View Detailed Metrics

Run the model evaluation script to see all metrics:

```bash
python model_evaluation.py
```

This will display a comprehensive evaluation report including:
- Training and testing metrics comparison
- Overfitting detection
- Interpretation guide for each metric

### Key Performance Indicators (KPIs)

**✅ Model is performing well if:**
- R² Score > 0.75 (explains >75% of variance)
- MAPE < 15% (predictions within ~15% of actual values)
- Test metrics close to training metrics (no overfitting)
- MAE small relative to the scale of EV numbers

---

## 🔧 Features Used for Prediction

The model uses the following engineered features for forecasting:

1. **Time-based Features**
   - `months_since_start`: Months elapsed since beginning of dataset

2. **County Information**
   - `county_encoded`: One-hot encoded county identifier

3. **Lag Features** (Previous EV counts)
   - `ev_total_lag1`: EV count from 1 month ago
   - `ev_total_lag2`: EV count from 2 months ago
   - `ev_total_lag3`: EV count from 3 months ago

4. **Statistical Features**
   - `ev_total_roll_mean_3`: Rolling average of EV counts (3 months)
   - `ev_total_pct_change_1`: Percentage change from 1 month ago
   - `ev_total_pct_change_3`: Percentage change from 3 months ago

5. **Growth Features**
   - `ev_growth_slope`: Linear trend slope of recent cumulative EV growth

---

## 📁 Project Structure

```
EV_Vehicle_Demand_Prediction/
├── EV_Forecasting.ipynb                # Main notebook with data exploration and model training
├── model_evaluation.py                 # Script to calculate and display accuracy metrics
├── app.py                              # Streamlit web application for predictions
├── preprocessed_ev_data.csv            # Cleaned and feature-engineered data
├── forecasting_ev_model.pkl            # Trained model file
├── evset.csv                           # Original raw dataset
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── ev_car.webp                         # UI image for web app
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pandas, numpy, scikit-learn, xgboost/lightgbm
- streamlit (for web app)
- joblib

### Installation

```bash
pip install -r requirements.txt
```

### Running the Evaluation

```bash
# View model accuracy metrics
python model_evaluation.py
```

### Running the Web Application

```bash
# Launch interactive forecasting tool
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

---

## 📈 Usage

### Interactive Forecasting Tool

1. Select a county from the dropdown
2. View historical EV adoption trend
3. See 3-year forecasted adoption
4. Compare trends across up to 3 counties
5. Understand growth percentages

### Programmatic Usage

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('forecasting_ev_model.pkl')

# Prepare features
features = {
    'months_since_start': 84,
    'county_encoded': 5,
    'ev_total_lag1': 2500,
    'ev_total_lag2': 2400,
    'ev_total_lag3': 2300,
    'ev_total_roll_mean_3': 2400,
    'ev_total_pct_change_1': 0.042,
    'ev_total_pct_change_3': 0.087,
    'ev_growth_slope': 45.2
}

# Make prediction
prediction = model.predict(pd.DataFrame([features]))
print(f"Predicted EV Count: {int(prediction[0])}")
```

---

## 📚 Model Interpretation

### Understanding the Metrics

- **R² Score (Coefficient of Determination)**
  - Range: 0 to 1
  - Interpretation: Percentage of variance in EV adoption explained by the model
  - Example: R² = 0.85 means model explains 85% of adoption variation

- **RMSE (Root Mean Squared Error)**
  - Unit: Number of vehicles
  - Interpretation: Average prediction error (penalizes large errors)
  - Useful for comparing predictions in the same units as target

- **MAE (Mean Absolute Error)**
  - Unit: Number of vehicles
  - Interpretation: Average magnitude of prediction errors
  - More robust to outliers than RMSE

- **MAPE (Mean Absolute Percentage Error)**
  - Unit: Percentage (%)
  - Interpretation: Average percentage deviation from actual values
  - Useful for comparing model performance across different scales

---

## 🎯 Next Steps & Improvements

1. **Model Enhancements**
   - Include additional features (EV charging station density, fuel prices)
   - Try ensemble methods combining multiple models
   - Implement auto-regressive ARIMA models

2. **Data Updates**
   - Incorporate real-time data feeds
   - Add external datasets (economic indicators, policy changes)

3. **Deployment**
   - Deploy web app to cloud (Heroku, AWS, Azure)
   - Create REST API for programmatic access
   - Build mobile application

4. **Monitoring**
   - Track model performance over time
   - Implement automated retraining pipeline
   - Set up alerts for prediction anomalies

---

## 👤 Author

**Hrishita Dey Purkayastha**

Prepared for the **AICTE Internship Cycle 2 by S4F**

---

## 📄 License

This project is open source and available for educational and research purposes.

---

## 📞 Questions & Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: hrishu071@gmail.com

---

**Last Updated**: June 2026
