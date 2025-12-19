


def predict_churn(data, model, threshold= 0.48950):
    pred_prob = model.predict_proba(data)
    churn_pred_prob = pred_prob[:, 1]
    churn = (churn_pred_prob >= threshold).astype(int)
    return churn_pred_prob, churn