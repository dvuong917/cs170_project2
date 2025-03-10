import numpy as np
import random

def nearest_neighbor(test_point, classes, features):
    min_dist = np.inf
    predicted_class = None
    for row, feature in enumerate(features): # compare test point features with dataSet features
        dist = distance(test_point, feature)
        if (dist < min_dist):
            min_dist = dist
            predicted_class = classes[row] # copy class of nearest neighbor
            print("New neighbor closest from row: ", row+1)
    return predicted_class

def distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

print("Welcome to Dylan Vuong's Feature Selection Algorithm.")
# inputFile = input("Type in the name of the file to test: ")
inputFile = "CS170_Small_Data__20.txt"
array = np.loadtxt(inputFile)
classes = array[:, 0].astype(int) # first column
features = array[:, 1:] # the rest of the columns
numRows = array.shape[0]
numColumns = array.shape[1]
print("numRows = ", numRows)
print("numColumns = ", numColumns)

def leave_one_out_cross_validation(data, current_set, feature_to_add): # testing stub
    return random.randint(0,100)

def feature_search_demo(data, numColumns):
    current_set_of_features = [] # initialize empty set
    for i in range(1, numColumns):
        print("On the", i, "th level of the search tree")
        feature_to_add_at_this_level = []
        best_so_far_accuracy = 0
        for k in range(1, numColumns):
            if (k not in current_set_of_features):
                print("--Considering adding the", k, "feature")
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k+1)
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
        current_set_of_features.append(feature_to_add_at_this_level)
        print("On level", i, ", added feature", feature_to_add_at_this_level, "to current set")

feature_search_demo(array, numColumns)

# # Test distance 
# testset = np.array([[0,2,2]])
# testfeature = testset[:, 1:]
# testpoint = np.array([1, 4])
# print("Distance: ", distance(testpoint, testfeature))

# # Test nearest neighbor
# testpoint1 = np.array([2.3515754e+00,   1.0158324e+00,  -2.1784832e+00,   1.5897041e-01,  -6.6376666e-01,  -2.8055787e-02]) # from row 18
# testpoint2 = np.array([-1.1075231e+00,   1.2113223e+00,  -8.6869643e-01,  -1.1616665e+00,  -1.4904901e+00,  -1.1996686e+00]) # from row 1
# print("(SHOULD RETURN 2) Predicted class: ", nearest_neighbor(testpoint1, classes, features))
# print("(SHOULD RETURN 2) Predicted class: ", nearest_neighbor(testpoint2, classes, features))

