from read import read_file
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import linear_model
import matplotlib.pyplot as plt 

def normalize(data):
    data = data.T # gets the transpose of a matrix
    for i in range(len(data)):
        data[i] = preprocessing.scale(data[i])
    data = data.T
    return data

def pca(data):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data)
    return principalComponents

def split_data(data):
    train, test = train_test_split(data, train_size=0.7)
    test_health = [i.pop() for i in test]
    train_health = [i.pop() for i in train]
    [i.pop() for i in test] # removing expected health
    [i.pop() for i in train]
    return train, test, train_health, test_health

def calculate_error(prediction, real):
    error = 0
    for i in range(len(real)):
        error += abs(prediction[i] - real[i])
    return error

def linear_regression(train, target, test):
    regr = linear_model.LinearRegression()
    regr.fit(train, target)

    prediction = regr.predict(test)
    return prediction


if __name__ == "__main__":
    np.random.seed(0)
    info = read_file()
 
    train, test, train_health, test_health = split_data(info)
    
    train = np.array(train)
    test = np.array(test)
    train = normalize(train)
    test = normalize(test)
    
    test_health = [i/max(test_health) for i in test_health]
    train_health = [i/max(train_health) for i in train_health]

    predicted_health = linear_regression(train, train_health, test)
    a = calculate_error(predicted_health, test_health)     

    print(a/sum(test_health))


    train = np.array(train)
    test = np.array(test)
    train = normalize(train)
    
    x = pca(train)
    y = pca(test)
    
    predicted_health = linear_regression(x, train_health, y)
    a = calculate_error(predicted_health, test_health)
    print(a/sum(test_health))    
