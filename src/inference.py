import joblib
from src import preprocessing


def load_models():
    all_models = []
    try:
        initial_columns = joblib.load("models/initial_columns.pkl")
        all_models.append(initial_columns)
    except:
        print("Initial Columns not found at {models/initial_columns.pkl}")
    try:
        model_one_hot_encoder = joblib.load("models/one_hot_encoder.pkl")
        all_models.append(model_one_hot_encoder)
    except:
        print("Model OneHotEncoder not found at {models/one_hot_encoder.pkl}")
    try: 
        tuned_logreg = joblib.load("models/tuned_logistic_regression.pkl")
        all_models.append(tuned_logreg)
    except:
        print("Tuned LogReg not found at {models/tuned_logistic_regression.pkl}")
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


def preprocess_customer_data(sample_data, model_one_hot_encoder, model_scaler, model_kmeans, model):
    df, num_cols, cat_cols = preprocessing.preprocess_df(sample_data)
    df['is_high_spender'] = df['MonthlyCharges'] > 70
    df['is_new_customer'] = df['tenure'] < 20
    services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', \
            'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies']
    service_count = 0
    for service in services:
        if service == 'InternetService': 
            service_count += df[service].apply(lambda x: 0 if x == 'No' else 1)
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
        if col != 'SeniorCitizen':
            df[col] = df[col].map(encoding_map).fillna(0)
    ohe_features = model_one_hot_encoder.transform(df[multi_cols])
    df[model_one_hot_encoder.get_feature_names_out()] = ohe_features.toarray()
    cols_to_drop = ['customerID', 'TotalCharges', 'avg_price', 'increase_amount', 'Churn']
    df = df.drop(multi_cols+cols_to_drop, axis=1)
    fixed_cols = [str(col).replace(" ", "_").replace("(", "").replace(")", "") for col in df.columns]
    df.columns = fixed_cols

    missing_columns = [col for col in df.columns if col not in model.feature_names_in_]
    try:
        assert (model.feature_names_in_ == df.columns).all()
    except:
        return Exception(f"Features names do not match!!! \n Your dataset has the columns, {missing_columns} which were not present during model fitting.")
    return df


def predict_churn(data, model, threshold= 0.48950):
    pred_prob = model.predict_proba(data)
    model_confidence = pred_prob.max()
    churn_pred_prob = pred_prob[:, 1]
    churn = (churn_pred_prob >= threshold).astype(int)
    return churn, churn_pred_prob, pred_prob, model_confidence