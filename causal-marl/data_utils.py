import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_data(df):
    scalers = {}
    df_norm = df.copy()

    for col in df.columns:
        scaler = StandardScaler()
        df_norm[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler

    return df_norm, scalers


def denormalize(pred_tensor, scaler):
    pred_np = pred_tensor.detach().cpu().numpy().reshape(-1, 1)
    return scaler.inverse_transform(pred_np).flatten()