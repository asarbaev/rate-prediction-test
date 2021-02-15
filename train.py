import pandas as pd
import numpy as np
import datetime  as dt

import scipy.stats as stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import matplotlib.pyplot as plt
%matplotlib inline

from model import Model

def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)

    return(res)

def to_labels(original_dataframe, feature_to_encode):
    label_encoder = preprocessing.LabelEncoder()
    original_dataframe[feature_to_encode] = label_encoder.fit_transform(original_dataframe[[feature_to_encode]])
    
    return(original_dataframe)

def drop_feature(dataset, feature):
    
    return(dataset.drop([feature], axis = 1))

def to_standardize(dataset, feature):
    scaler = preprocessing.StandardScaler()
    dataset[feature] = scaler.fit_transform(dataset[[feature]])
   
    return(dataset)

def to_normalize(dataset, feature):
    scaler = preprocessing.MinMaxScaler()
    dataset[feature] = scaler.fit_transform(dataset[[feature]])
   
    return(dataset)
    
def data_preprocessing(dataset):
    
    dataset['log_rate'] = np.log(dataset['rate'])
    dataset['log_miles'] = np.log1p(dataset['valid_miles']) 
    
    dataset['isTheSameLocation'] = np.where(dataset['origin_kma'] == dataset['destination_kma'], 1, 0) 
    
    dataset['month'] = pd.DatetimeIndex(dataset['pickup_date']).month
    dataset['day'] = pd.DatetimeIndex(dataset['pickup_date']).day
    dataset['time'] = pd.DatetimeIndex(dataset['pickup_date']).hour * 60 + pd.DatetimeIndex(dataset['pickup_date']).minute
    
    kmeans = MiniBatchKMeans(n_clusters = 8, batch_size=1000, random_state = 0).fit(dataset[['weight']])
    dataset['weight_cluster'] = kmeans.predict(dataset[['weight']])
    
    #labeling for categorical features
    features_to_label = ['transport_type', 'origin_kma', 'destination_kma']
    for feature in features_to_label:
        dataset = to_labels(dataset, feature)

    #remove useless features
    features_to_drop = ['pickup_date', 'rate', 'valid_miles']
    for feature in features_to_drop:
        dataset = drop_feature(dataset, feature)
        
    #normalization
    features_to_normalize = ['log_rate', 'log_miles', 'weight', 'time']
    for feature in features_to_normalize:
        dataset = to_normalize(dataset, feature)
        
    dataset[['transport_type', 'origin_kma', 'destination_kma', 'weight_cluster']] = \
    dataset[['transport_type', 'origin_kma', 'destination_kma', 'weight_cluster']].astype('string')
    
    #one-hot encoding for the categorical features
    features_to_encode = ['transport_type', 'origin_kma', 'destination_kma', 'weight_cluster']
    for feature in features_to_encode:
        dataset = encode_and_bind(dataset, feature)
    
    return(dataset)

def accuracy(real_rates, predicted_rates):
    return np.average(abs(predicted_rates / real_rates - 1.0)) * 100.0

def train_and_validate(training_dataset_path, validation_dataset_path):
     
    training_dataset = pd.read_csv(training_dataset_path)
    validation_dataset = pd.read_csv(validation_dataset_path)
    
    training_dataset['weight'] = training_dataset['weight'].fillna(training_dataset['weight'].mean())
    
    #variable to store a break between merged training + validation sets and test set
    training_size = training_dataset.shape[0]
    df = training_dataset.append(validation_dataset).reset_index(drop=True)
    
    df = data_preprocessing(df)
    
    df_train = df.iloc[:training_size, :]
    Y_train = df_train[['log_rate']]
    X_train = drop_feature(df_train, 'log_rate')

    df_val = df.iloc[training_size:, :]
    Y_val = df_val[['log_rate']]
    X_val = drop_feature(df_val, 'log_rate')
    
    # create a regressor object 
    regressor = RandomForestRegressor(random_state = 0, n_jobs = -1, n_estimators = 100, verbose = 2)
    regressor.fit(X_train.values, Y_train.values.ravel())
    
    predicted_rates = regressor.predict(X_val.values)
    
    mare = accuracy(Y_val.values.ravel(), predicted_rates)
    mare = np.round(mare, 2)
    return mare

def generate_final_solution(model, test_dataset):
    test_dataset = drop_feature(test_dataset, 'rate')
    predictions = model.predict(test_dataset.values)
    print(predictions)

if __name__ == "__main__":
    mare = train_and_validate('dataset/train.csv', 'dataset/validation.csv')
    print("Accuracy of validatTion is %s" %(mare))
    
