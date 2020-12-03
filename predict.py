from os import path
import joblib
import pandas as pd
import numpy as np


def predict(df):
    model_file = 'artifacts/model.joblib'
    if not path.exists(model_file):
        raise FileNotFoundError(f'Model artifacts not found, please run "train.py" first to create {model_file}')

    mean_rate = joblib.load(model_file)
    return [mean_rate] * len(df)


def accuracy(real_rates, predicted_rates):
    return np.average(abs(real_rates / predicted_rates - 1.0)) * 100.0


if __name__ == "__main__":
    df = pd.read_csv('dataset/validation.csv')
    predicted_rates = predict(df)
    mare = accuracy(df.rate, predicted_rates)
    mare = np.round(mare, 2)
    print(f'Accuracy of prediction is {mare}%')
