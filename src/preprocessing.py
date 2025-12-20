import pandas as pd
import numpy as np

def preprocess_df(df:pd.DataFrame):
    total_charges = []
    for val in df['TotalCharges']:
        try:
            total_charges.append(float(val))
        except ValueError:
            total_charges.append(0)
    df['TotalCharges'] = total_charges
    df.loc[df['TotalCharges']<1, 'TotalCharges'] = df['TotalCharges'].median()

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    cat_cols.extend(['SeniorCitizen'])
    cat_cols.remove('customerID')
    cat_cols.remove('Churn')
    num_cols = df.select_dtypes(include='number').columns.tolist()
    num_cols.remove('SeniorCitizen')
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(object)
    # df[]
    return df, num_cols, cat_cols