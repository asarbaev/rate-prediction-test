import joblib
import pandas as pd
import numpy as np
from os import path


def train(df):
    mean_rate = df.rate.mean()
    print('Average rate:', mean_rate)
    joblib.dump(mean_rate, 'artifacts/model.joblib')


def predict(df):
    model_file = 'artifacts/model.joblib'
    if not path.exists(model_file):
        raise FileNotFoundError(f'Model artifacts not found, please run "train.py" first to create {model_file}')

    mean_rate = joblib.load(model_file)
    return [mean_rate] * len(df)


def accuracy(real_rates, predicted_rates):
    return np.average(abs(predicted_rates / real_rates - 1.0)) * 100.0


def train_and_validate():
    df = pd.read_csv('dataset/train.csv')
    train(df)

    df = pd.read_csv('dataset/validation.csv')
    predicted_rates = predict(df)
    mare = accuracy(df.rate, predicted_rates)
    mare = np.round(mare, 2)
    return mare


def generate_final_solution():
    # combine train and validation to improve final predictions
    df = pd.read_csv('dataset/train.csv')
    df_val = pd.read_csv('dataset/validation.csv')
    df = df.append(df_val).reset_index(drop=True)
    train(df)

    # generate and save test predictions
    df_test = pd.read_csv('dataset/test.csv')
    df_test['predicted_rate'] = predict(df_test)
    df_test.to_csv('dataset/predicted.csv', index=False)


if __name__ == "__main__":
    mare = train_and_validate()
    print(f'Accuracy of validation is {mare}%')

    if mare < 13:  # try to reach 13% or less for validation
        generate_final_solution()
        print("'predicted.csv' is generated, please send it to us")
