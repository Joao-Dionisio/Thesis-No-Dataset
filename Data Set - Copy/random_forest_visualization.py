from sklearn.tree import export_graphviz
import pydot


def visualize(rf, feature_list):

    tree = rf.estimators_[5]
    
    # Export the image to a dot file
    export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
    
    # Use dot file to create a graph
    (graph, ) = pydot.graph_from_dot_file('tree.dot')
    
    # Write graph to a png file
    graph.write_png('tree.png')
    
    return     


def feature_importance(rf, feature_list):

    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    
    # Print out the feature and importances 
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];