from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
from random_forest_visualization import visualize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


def random_forest_regression(df, target):
    features = df
    
    # The target variable
    labels = np.array(features[target])
    
    # The rest of the variables
    features = features.drop(target, axis = 1)
    
    # For later use
    feature_list = list(features.columns)
    
    # Split test and train
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 37)

    # Build Random Forest
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42) # We can make the trees smaller with max_depth = k.
    
    # Fit Random Forest with test data
    rf.fit(train_features, train_labels)

    # Make predictions
    predictions = rf.predict(test_features)
    predictions = (predictions - predictions.mean())/predictions.std()
    test_labels = (test_labels - test_labels.mean())/test_labels.std()
    # Calculate accuracy from mean absolute percentage error
    errors = abs(predictions - test_labels)
    mape = 100 * (errors/ test_labels)
    accuracy = 100 - np.mean(mape)

    #print('Accuracy:', round(np.mean(errors), 2))
    print('Accuracy:', accuracy)
    #a = np.sum(predictions == test_labels)
    #print(test_labels)
    #print(predictions)

    #for i in range(len(predictions)):
    #    print(test_labels[i]-min(test_labels), predictions[i]-min(test_labels))
    #TP, FP, TN, FN = perf_measure(test_labels, predictions)
    #print("Recall:", TP/(TP+FN))
    #print("Precision:", TP/(TP+FP))
    #print("Accuracy:", (TP+TN)/(TP+TN+FP+FN))


def random_forest_classification(df, target):
    features = df
    
    # The target variable
    labels = np.array(features[target])
    
    # The rest of the variables
    features = features.drop(target, axis = 1)
    
    # For later use
    feature_list = list(features.columns)
    
    # Split test and train
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 37)

    weights = {0:1, 1:14}    
    #weights = {0:1, 1:1}
    clf = RandomForestClassifier(max_depth=2, random_state=73, class_weight = weights)
    clf.fit(train_features, train_labels)

    predictions = clf.predict(test_features)

    TP, FP, TN, FN = perf_measure(test_labels, predictions)
    
    print("Recall:", TP/(TP+FN))
    print("Precision:", TP/(TP+FP))
    print("Accuracy:", (TP+TN)/(TP+TN+FP+FN))

    return df


def perf_measure(y_actual, y_hat):
    # https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)
    
if __name__ == "__main__":
    random_forest_classifier(df, "Ano Avaria")
