import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def binary_enc(df:pd.DataFrame):
    df['is_high_spender'] = df['MonthlyCharges'] > 70
    df['is_new_customer'] = df['tenure'] < 20
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

def service_count(df:pd.DataFrame):
    """Count the number of services a customer has subscribed to"""
    services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', \
            'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies']
    service_count = 0
    for service in services:
        if service == 'InternetService': 
            service_count += df[service].apply(lambda x: 0 if x == 'No' else 1)
        else:
            service_count += df[service].apply(lambda x: 1 if x== 'Yes' else 0)
    df['service_count'] = service_count
    return df

def check_price_increase(df:pd.DataFrame):
    """Check if the customer's montly charges have changed"""
    df['tenure'] = df['tenure'].replace(0,1)
    df ['avg_price'] = round((df['TotalCharges']/df['tenure']), 2)
    df['increase_amount'] =  round((df['MonthlyCharges'] - df['avg_price']), 2)
    df['increase_pct'] = round(((df['increase_amount']/df['MonthlyCharges']) *100), 2)
    return df

def cluster_customers(df:pd.DataFrame, n_clusters:int =4, visualize:bool = False):
    """Cluster customers based on tenure, Monthly charges and Total charges"""
    cols_to_cluster = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[cols_to_cluster])
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    customer_segments = kmeans.fit_predict(scaled_features)
    df['customer_segment'] = customer_segments
    if visualize:
        sns.scatterplot(x=scaled_features[:, 0], y=scaled_features[:, 1], c=kmeans.labels_)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='X', s=100, label='Centroids')
        plt.axis('off')
        plt.suptitle(f"Customers Segmentation")
        return 
    return df, scaler, kmeans

def encode_cat_cols(df, cat_cols):
    """Encode categorical features to make them ready to be fed into our models"""
    encoding_map = {'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0}
    multi_cols = ['InternetService', 'Contract', 'PaymentMethod']
    binary_cols = [c for c in cat_cols if c not in multi_cols]
    for col in binary_cols:
        if col != 'SeniorCitizen':
            df[col] = df[col].map(encoding_map).fillna(0)
    ohe = OneHotEncoder(drop='first')
    ohe_features = ohe.fit_transform(df[multi_cols])
    ohe_columns = ohe.get_feature_names_out()
    df[ohe_columns] = ohe_features.toarray()
    df = df.drop(multi_cols, axis=1)
    # df = pd.get_dummies(df, columns=multi_cols, drop_first=True)    
    fixed_cols = [col.replace(" ", "_").replace("(", "").replace(")", "") for col in df.columns]
    df.columns = fixed_cols
    return df, ohe
