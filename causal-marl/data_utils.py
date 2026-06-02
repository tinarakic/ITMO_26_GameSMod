import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_data(df):
    """
    Нормализует числовые столбцы DataFrame с помощью z-меток (StandardScaler).

    Для каждого столбца создается отдельный StandardScaler, который вычисляет
    среднее и стандартное отклонение. Значения преобразуются по формуле:
        x_norm = (x - mean) / std

    Args:
        df: pandas.DataFrame с числовыми признаками для нормализации

    Returns:
        tuple: (df_norm, scalers)
            df_norm: pandas.DataFrame с нормализованными данными
            scalers: dict {column_name: StandardScaler} для последующего денормирования
    """
    scalers = {}
    df_norm = df.copy()

    for col in df.columns:
        scaler = StandardScaler()
        df_norm[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler

    return df_norm, scalers


def denormalize(pred_tensor, scaler):
    '''
    Восстанавливает исходный масштаб данных после нормализации.
    
    Args:
        pred_tensor: torch.Tensor с нормализованными предсказаниями
        scaler: sklearn.preprocessing.StandardScaler, использованный для нормализации соответствующего столбца

    Returns:
        numpy.ndarray: массив предсказаний в исходных единицах измерения
    '''
    pred_np = pred_tensor.detach().cpu().numpy().reshape(-1, 1)
    return scaler.inverse_transform(pred_np).flatten()