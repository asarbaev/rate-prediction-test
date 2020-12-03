import joblib
import pandas as pd


def train(df):
    mean_rate = df.rate.mean()
    print('Average rate:', mean_rate)
    joblib.dump(mean_rate, 'artifacts/model.joblib')


if __name__ == "__main__":
    df = pd.read_csv('dataset/train.csv')
    train(df)
    print("Train completed, run 'predict.py' now")
