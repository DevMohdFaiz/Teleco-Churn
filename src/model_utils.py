import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,\
                             classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, make_scorer, fbeta_score
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import SMOTE




def split_data(df, cols_to_drop= ['customerID', 'TotalCharges', 'avg_price', 'increase_amount', 'Churn']):
    X = df.drop(cols_to_drop, axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=50)
    return X_train, X_test, y_train, y_test 


def train_model(model, data):
    model_name = re.match(r"\w+", model.__str__()).group()
    X_train, X_test, y_train, y_test = data
    if not hasattr(model, 'n_features_in_'):
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]
    print("="*10)
    print(f"{model_name}")
    print(f"Train: {model.score(X_train, y_train)} | Test: {model.score(X_test, y_test)} | Accuracy: {accuracy_score(y_test, y_pred)}")
    return model, y_pred, pred_proba

def tune_logistic_regression(model, X_train, y_train):
    param_grid={
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'], 
        # 'max_iter': [2000, 3000, 5000, 8000],
        'class_weight': [None, 'balanced', {0:1, 1:2}, {0:1, 1:3}]
    }

    f2_scorer = make_scorer(fbeta_score, beta=2)
    logreg_grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=f2_scorer, cv=5, verbose=3, n_jobs=-1)
    logreg_grid.fit(X_train, y_train)
    return logreg_grid

def evaluate_model(model, y_test, y_pred, y_pred_proba, plot_confusion_matrix=False):
    model_name = re.match(r"\w+", model.__str__()).group()
    print("="*15)
    print(f"{model_name}")
    print(f"ROC AUC SCORE: {roc_auc_score(y_test, y_pred_proba):.3f}")
    print(f"\nClassification report\n")
    print(classification_report(y_test, y_pred))

    if plot_confusion_matrix:
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt=".0f", cmap='Blues')
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.yticks(rotation=0)
        plt.title(f"{model_name} confusion matrix");
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return recall, precision, f1

def plot_roc_curve(model, y_test, y_pred_proba):
    model_name = re.match(r"\w+", model.__str__()).group()
    fpr, tpr, _ =roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.legend([f"Model: roc_auc: {roc_auc_score(y_test, y_pred_proba):.2f}", "Random guess"])
    plt.title(f"{model_name} ROC-Curve")

def plot_feature_importance(model):    
    model_name = re.match(r"\w+", model.__str__()).group()
    try:
        feat_vals = model.feature_importances_ if hasattr(model, 'feature_importances_') else model.coef_[0]
    except:
        return f"{model_name} has no feature importance attribute"
    feat_vals_list = feat_vals.tolist()
    feat_names = model.feature_names_in_
    feat_names_list = feat_names.tolist()
    feat_imp_df = pd.DataFrame(feat_vals_list, index=feat_names_list, columns=['Importance'])
    feat_imp_df.sort_values('Importance', ascending=False, inplace=True)
    sns.barplot(x=feat_imp_df.Importance, y=feat_imp_df.index)
    plt.title(f'{model_name} Feature Importances')
    plt.ylabel('Feature');
    return feat_imp_df

def plot_precision_recall(model, y_test, y_pred_proba, recall_threshold=.75, chosen_threshold=0.48950):
    model_name = re.match(r"\w+", model.__str__()).group()
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    precisions_list, recalls_list, thresholds_list  = precisions[:-1].tolist(), recalls[:-1].tolist(), thresholds.tolist()
    trade_df = pd.DataFrame({'threshold': thresholds_list, 'precision': precisions_list, 'recall': recalls_list})
    best_trade_df = trade_df[trade_df['recall']>=recall_threshold].sort_values('precision', ascending=False).head(300)
    plt.plot(best_trade_df['threshold'], best_trade_df['recall'], label='Recall')
    plt.plot(best_trade_df['threshold'], best_trade_df['precision'], label='Precision')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.axvline(x=chosen_threshold, color='green', linestyle='--', label='My chosen threshold')
    # plt.axline(y=.56)
    plt.yticks(np.arange(.45, .91, .05))
    plt.legend()
    plt.grid()
    plt.title(f"{model_name} Precision-Recall Curve")
    return best_trade_df


def anaylze_sacrificed_features(dead_feat, logreg_feat_imp, X_train):
    """See which features killed off another with L1 (lasso) in Logistic Regression"""
    # dead_feat = "InternetService_Fiber optic" #logreg_feat_imp[logreg_feat_imp['Importance']==0].index.tolist()
    survivor_feat = logreg_feat_imp[logreg_feat_imp['Importance']!=0].index.tolist()
    feat_corr =X_train[[dead_feat]+survivor_feat].corr()[dead_feat]
    feat_corr=feat_corr.abs().sort_values(ascending=False).drop(dead_feat)
    result =(f"{dead_feat} was sacrified for {feat_corr.index[0]} with a correlation of {feat_corr.values[0]:.2f}") if len(dead_feat.values) is not None else f"No dead features found!"
    return result