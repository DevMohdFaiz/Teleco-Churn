import joblib
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src import preprocessing


class CustomerData(BaseModel):
    tenure: int
    gender: str
    SeniorCitizen: str
    Partner: str
    Dependents: str
    PhoneService: object
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    # gender: int
    # SeniorCitizen: int
    # Partner: int
    # Dependents: int
    # tenure:  int
    # PhoneService: int
    # MultipleLines: int
    # OnlineSecurity: int
    # OnlineBackup: int
    # DeviceProtection: int
    # TechSupport: int
    # StreamingTV: int
    # StreamingMovies: int
    # PaperlessBilling: int
    # MonthlyCharges: float
    # is_high_spender: bool
    # is_new_customer: bool
    # service_count: int
    # increase_pct: float
    # customer_segment: int
    # InternetService_Fiber_optic: bool
    # InternetService_No: bool
    # Contract_One_year: bool
    # Contract_Two_year: bool
    # PaymentMethod_Credit_card_automatic: bool
    # PaymentMethod_Electronic_check: bool
    # PaymentMethod_Mailed_check: bool


class PredictionResponse(BaseModel):
    prediction: int
    non_churn_prob: float
    churn_prob: float
    risk_level: str
    # model_confidence: list #float



def load_models():
    all_models = []
    try:
        initial_columns = joblib.load("models/initial_columns.pkl")
        all_models.append(initial_columns)
    except:
        print("Initial Columns not found at {models/initial_columns.pkl}")

    try: 
        tuned_logreg = joblib.load("models/tuned_logistic_regression.pkl")
        all_models.append(tuned_logreg)
    except:
        print("Tuned LogReg not found at {models/tuned_logistic_regression.pkl}")

    try:
        model_one_hot_encoder = joblib.load("models/one_hot_encoder.pkl")
        all_models.append(model_one_hot_encoder)
    except:
        print("Model OneHotEncoder not found at {models/one_hot_encoder.pkl}")

    try: 
        model_scaler= joblib.load("models/model_scaler.pkl")
        all_models.append(model_scaler)
    except:
        print("Model Scaler not found at {models/model_scaler.pkl}")

    try: 
        model_kmeans = joblib.load("models/model_kmeans.pkl")
        all_models.append(model_kmeans)
    except:
        print("Model Kmeans not found at {models/model_kmeans.pkl}")
        
    if len(all_models)<5:
        return f"Pls check again and ensure all five models are properly loaded!"
    else:
        return all_models


def preprocess_customer_data(customer_data, model, model_one_hot_encoder, model_scaler, model_kmeans):
    df = pd.DataFrame([customer_data.model_dump()])
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    cat_cols.extend(['SeniorCitizen'])
    num_cols = df.select_dtypes(include='number').columns.tolist()
    df['is_high_spender'] = df['MonthlyCharges'] > 70
    df['is_new_customer'] = df['tenure'] < 20
    services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', \
            'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies']
    service_count = 0
    for service in services:
        if service == 'InternetService': 
            service_count += df[service].apply(lambda x: 0 if x == 'No' else 1)
        else:
            service_count += df[service].apply(lambda x: 1 if x== 'Yes' else 0)
    df['service_count'] = service_count

    df['tenure'] = df['tenure'].replace(0,1)
    df ['avg_price'] = round((df['TotalCharges']/df['tenure']), 2)
    df['increase_amount'] =  round((df['MonthlyCharges'] - df['avg_price']), 2)
    df['increase_pct'] = round(((df['increase_amount']/df['MonthlyCharges']) *100), 2)
    cols_to_cluster = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaled_features = model_scaler.transform(df[cols_to_cluster])
    customer_segment = model_kmeans.predict(scaled_features)
    df['customer_segment'] = customer_segment
    encoding_map = {'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0}
    multi_cols = ['InternetService', 'Contract', 'PaymentMethod']
    binary_cols = [c for c in cat_cols if c not in multi_cols]
    for col in binary_cols:
        df[col] = df[col].map(encoding_map).fillna(0)
    ohe_features = model_one_hot_encoder.transform(df[multi_cols])
    df[model_one_hot_encoder.get_feature_names_out()] = ohe_features.toarray()
    cols_to_drop = ['TotalCharges', 'avg_price', 'increase_amount']
    df = df.drop(multi_cols+cols_to_drop, axis=1)
    fixed_cols = [str(col).replace(" ", "_").replace("(", "").replace(")", "") for col in df.columns]
    df.columns = fixed_cols
    df = df.reindex(columns=model.feature_names_in_)
    missing_columns = [col for (idx, col) in enumerate(df.columns) if col != model.feature_names_in_[idx]]
    try:
        assert (model.feature_names_in_ == df.columns).all()
    except:
        return Exception(f"Features names do not match!!! \n Your dataset has the columns, {missing_columns} which were not present during model fitting.")
    return df

app =FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins= ["*"],
    allow_credentials =True,
    allow_methods= ["*"],
    allow_headers = ["*"]
)
models = {}

@app.on_event("startup")
async def startup_event():
    print("Loading models...")
    try:
        (
            models["initial_columns"],
            models["model"],  
            models["model_one_hot_encoder"], 
            models["model_scaler"], 
            models["model_kmeans"]
        ) = load_models()
        print(f"Loaded all models...")
    except Exception as e:
        print(f"An Error occured: {e}")


@app.get("/")
def home():
    return {"message": "Welcome to the Churn Prediction API"}

@app.get("/health")
def get_health():
    return {"status": "healthy", "n_models": len(models)}

@app.post("/predict")
def predict_churn(customer_data: CustomerData, threshold= 0.48950):
    # initial_columns, model, model_one_hot_encoder, model_scaler, model_kmeans = inference.load_models()
    customer_df = preprocess_customer_data(
                            customer_data,
                            model=models['model'],
                            model_one_hot_encoder=models['model_one_hot_encoder'], 
                            model_scaler=models['model_scaler'],
                            model_kmeans=models['model_kmeans']
                    )
    pred_prob = models['model'].predict_proba(customer_df)
   
    non_churn_prob = float(pred_prob[0, 0])     
    churn_probability = float(pred_prob[0, 1]) 
    prediction = int(churn_probability >= threshold)
    if churn_probability >= 0.7:
        risk_level = 'high'
    elif churn_probability >=0.4 and churn_probability < 0.7:
        risk_level = 'medium'
    else:
        risk_level = 'low'
    return PredictionResponse(
        prediction= prediction, non_churn_prob= non_churn_prob, churn_prob=churn_probability, risk_level = risk_level
    ) 


if __name__== '__main__':
    print(f"Running FastAPI")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)