
# ⚡ Short-Term Electricity Demand Forecasting in the UK Using Machine Learning with Weather Integration

# 📌 Overview
A machine learning and deep learning–based forecasting system that predicts **daily UK electricity demand** using historical demand and weather data.  
The project includes an **interactive Streamlit dashboard** for:
- Forecasting
- Visualization
- Model evaluation

# 🧠 Models Used
- **XGBoost Multivariate Model**
- **LSTM Multivariate Model**

# 🖥️ Dashboard Features
- Select forecast date  
- Choose prediction model  
- Display predicted electricity demand  
- Download forecast as CSV  
- View model evaluation metrics  
- Feature importance visualization  
- Actual vs Predicted comparison  
- Seasonal error analysis  

# 📂 Project Structure
```
UK-Electricity-Demand-Forecasting/

├── app.py  
├── data/  
├── models/  
├── notebooks/  
├── requirements.txt  
└── README.md  
```

# ⚙️ Installation

## 🔹 Create Virtual Environment

```
python -m venv weather_env --without-pip
```

## 🔹 Activate Virtual Environment

```
weather_env\Scripts\activate
```

## 🔹 Upgrade pip

```
python -m pip install --upgrade pip
```

## 🔹 Install Required Libraries

```
pip install -r requirements.txt
```

## 🔹 Freeze Installed Libraries

```
pip freeze > requirements.txt
```

# 🚀 Run the Application

```
python -m streamlit run main.py
```

Open in browser:

```
http://localhost:8501
```

# 📊 Dataset Information

- **Frequency:** Daily  
- **Target Variable:** `daily_demand (MW)`  

Includes:

- Historical demand  
- Weather features  
- Lag features  
- Calendar variables  

# 📦 Dependencies

```
streamlit
pandas
numpy
scikit-learn
tensorflow
matplotlib
xgboost
joblib
```

# 🎯 Use Cases

- Electricity demand forecasting  
- Power grid planning  
- Load forecasting  
- Energy demand analysis  

# 🔮 Future Improvements

- Real-time weather API integration  
- Multi-step forecasting  
- Cloud deployment  
- REST API integration  

# 👤 Author

**Abdul Sami**  
**M.Sc Data Science & Artificial Intelligence**  
**Student ID:** 2414347
