import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from typing import List, Tuple



def plot_churn_distribution(df):
    """Plot the churn distribution in a Pie Chart"""
    churn_vals = df['Churn'].value_counts(normalize=True)*100
    fig = px.pie(values=churn_vals.values, names=['No Churn', 'Churn'], labels=['No Churn', 'Churn'], hole=.3)
    fig.update_traces(textposition='inside', textinfo='label+percent')
    fig.update_layout({'showlegend': False, 'title':'Churn Distribution'})
    fig.show()


def plot_cat_cols(df: pd.DataFrame, cat_cols: List, hue: bool=False):
    nrows= 6
    ncols = int(np.ceil(len(cat_cols)/nrows))
    fig, ax = plt.subplots(figsize=(20, 17), nrows=nrows, ncols=ncols, constrained_layout=True)
    ax = ax.flatten()
    fig.suptitle("Categorical Columns")
    for idx, col in enumerate(cat_cols):
        hue = 'Churn' if hue else None
        axis = sns.countplot(df, x=df[col], ax=ax[idx], hue=hue)
        axis.set_in_layout(in_layout=True)
        axis.set_title(f"{col}")
        axis.set_ylabel('')
        axis.set_xlabel('')
        axis.tick_params(axis='x', rotation=25)
        axis.bar_label(axis.containers[0], fmt= lambda x: f"{x/len(df[col])*100:.2f}%")
        axis.set_ylim(0, df[col].value_counts().values[0]*1.1)
        # axis.get_tightbbox(fig)
    
    for j in range(idx+1, len(ax)):
        fig.delaxes(ax[j]) 
        

def plot_num_cols(df, num_cols: List, hue:bool=False, kde=False, nrows=1):
    ncols = int(np.ceil(len(num_cols)/nrows))
    fig, ax = plt.subplots(figsize=(15, 4), nrows=nrows, ncols=ncols, constrained_layout=True)
    ax = ax.flatten()
    fig.suptitle("Numerical Columns")
    # fig.tight_layout(h_pad=50, w_pad=40)
    hue = 'Churn' if hue else None
    for idx, col in enumerate(num_cols):
        axis = sns.kdeplot(df, x=df[col], ax=ax[idx], hue=hue) if kde else sns.histplot(df, x=df[col], ax=ax[idx], kde=True, hue=hue) 
        axis.set_in_layout(in_layout=True)
        axis.set_title(f"{col}")
        axis.set_ylabel('')
        axis.set_xlabel('')
    
    for j in range(idx+1, len(ax)):
        fig.delaxes(ax[j])


def plot_churn_ratio(df: pd.DataFrame, cat_cols: List, nrows:int =3, figsize: Tuple =(15, 7)):
    ncols = int(np.ceil(len(cat_cols)/nrows))
    fig, ax = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, constrained_layout=True)
    fig.suptitle(f"Churn Ratio")
    ax= ax.flatten()
    for idx, col in enumerate(cat_cols):
            churn_df = pd.crosstab(df[col], df['Churn'])
            churn_df['churn_ratio'] = churn_df['Yes']/ (churn_df['No']+ churn_df['Yes'])
            axis = sns.barplot(churn_df['churn_ratio'], ax=ax[idx])
            axis.set_title(f"{col}")
            axis.set_xlabel('')
            axis.set_ylabel('')
            rotation = 0 if len(churn_df) <=3 else 15
            axis.tick_params(axis='x', rotation=rotation)
            axis.bar_label(axis.containers[0], fmt=lambda x: f'{(x/len(churn_df))*100:0.2f}%')
            axis.set_ylim(0, (churn_df['churn_ratio'].max())*1.1)
    # plt.subplots_adjust(hspace=2, wspace=.5)

    for j in range(idx+1, len(ax)):
        fig.delaxes(ax[j])