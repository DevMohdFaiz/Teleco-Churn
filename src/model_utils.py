import re
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
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
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]
    print("="*10)
    print(f"{model_name}")
    print(f"Train: {model.score(X_train, y_train)} | Test: {model.score(X_test, y_test)} | Accuracy: {accuracy_score(y_test, y_pred)}")
    return model, y_pred, pred_proba

def evaluate_model(model, y_test, y_pred, plot_confusion_matrix=False):
    model_name = re.match(r"\w+", model.__str__()).group()
    print("="*15)
    print(f"{model_name} classification report")
    print(classification_report(y_test, y_pred))

    if plot_confusion_matrix:
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt=".0f", cmap='Blues')
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.yticks(rotation=0);
        plt.title(f"{model_name} confusion matrix")

def plot_roc_curve(model, y_test, y_pred, y_pred_proba):
    model_name = re.match(r"\w+", model.__str__()).group()
    fpr, tpr, _ =roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.legend([f"Model: roc_auc: {roc_auc_score(y_test, y_pred):.2f}", "Random guess"])
    plt.title(f"{model_name} ROC-Curve")

def plot_feature_importance(model, ax=None):    
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
    sns.barplot(x=feat_imp_df.Importance, y=feat_imp_df.index, ax=ax)
    plt.title(f'{model_name} Feature Importances')
    plt.ylabel('Feature');
    return feat_imp_df