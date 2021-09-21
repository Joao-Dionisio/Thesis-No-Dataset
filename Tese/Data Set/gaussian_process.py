from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
import numpy as np
from random_forest import perf_measure

def gaussian_process_classifier(df, target):
    kernel = 1.0 * RBF(1.0)
    
    features = df
    
    # The target variable
    labels = np.array(features[target])
    
    # The rest of the variables
    features = features.drop(target, axis = 1)
    
    # For later use
    feature_list = list(features.columns)
    
    # Split test and train
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 37)

    gpc = GaussianProcessClassifier(kernel=kernel,
             random_state=0).fit(train_features, train_labels)

    predictions = []
    probabilities = gpc.predict_proba(test_features)
    for i in probabilities:
        if i[0] < 0.90:
            predictions.append(1)
        else:
            predictions.append(0)

		
    # Calculate accuracy from mean absolute percentage error
    errors = abs(predictions - test_labels)
    mape = 100 * (errors/ test_labels)
    print(mape)
    accuracy = 100 - np.mean(mape)

    #print('Accuracy:', round(np.mean(errors), 2))
    print('Accuracy:', accuracy)
    #a = np.sum(predictions == test_labels)
    #print(test_labels)
    #print(predictions)
    
    TP, FP, TN, FN = perf_measure(test_labels, predictions)
    print("Recall:", TP/(TP+FN))
    print("Precision:", TP/(TP+FP))
    #print("Accuracy:", (TP+TN)/(TP+TN+FP+FN))

