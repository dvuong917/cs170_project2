import numpy as np
import time
# import random

def distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

# def leave_one_out_rand_stub(data, current_set, feature_to_add): # testing stub
#     return random.randint(0,100)

def leave_one_out_cross_validation(data, current_set, feature_to_add):
    numRows = data.shape[0]
    number_correctly_classified = 0

    if feature_to_add is None:
        selected_features = current_set  
    else:
        selected_features = current_set + [feature_to_add] # use brackets since current_set is list and feature_to_add is int

    # make local copy of dataset with ignored features
    data_copy = data[:, [0] + [x+1 for x in selected_features]] # +1 to avoid class label column

    for i in range(numRows): # change to numRows
        object_to_classify = data_copy[i, 1:]
        label_object_to_classify = data_copy[i, 0]

        nearest_neighbor_dist = np.inf

        for k in range(numRows): # change to numRows
            if k != i: # don't compare to yourself
                # print("Ask if", i, "is nearest neighbor with", k)
                # print("object to classify = ", object_to_classify)
                # print("other objects = ", other_objects)
                dist = distance(object_to_classify, data_copy[k, 1:])
                if (dist < nearest_neighbor_dist):
                    nearest_neighbor_dist = dist
                    nearest_neighbor_label = data_copy[k, 0]
        # print("Looping over i, at the", i, "location")
        # print("The", i, "th object is in class", classes)

        # print("Object", i+1, "is class", label_object_to_classify)
        # print("Its nearest_neighbor is", nearest_neighbor_loc+1, "which is in class", nearest_neighbor_label)

        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
    
    accuracy = number_correctly_classified / numRows
    # print(f"Selected features: {selected_features}")
    # print("Accuracy = ", accuracy*100, "%")
    return accuracy


def forward_feature_search_demo(data, numColumns):
    current_set_of_features = [] # initialize empty set
    overall_best_accuracy = 0
    overall_best_set = []

    base_accuracy = leave_one_out_cross_validation(data, [], None)
    rounded_base = round(base_accuracy * 100, 5)
    print(f"    Using feature(s) [], accuracy is {rounded_base}%\n")

    for i in range(numColumns):
        # print("On the", i+1, "th level of the search tree")
        feature_to_add_at_this_level = None
        best_so_far_accuracy = 0
        for k in range(numColumns):
            if (k not in current_set_of_features):
                # print("--Considering adding the", k+1, "feature")
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k)
                rounded_accuracy = round(accuracy * 100, 5)
                print(f"    Using feature(s) {[x+1 for x in current_set_of_features] + [k+1]}, accuracy is {rounded_accuracy}%")
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
        print()

        current_set_of_features.append(feature_to_add_at_this_level)
        if best_so_far_accuracy > overall_best_accuracy:
            overall_best_accuracy = best_so_far_accuracy
            overall_best_set = current_set_of_features.copy()
        else:
            print("(Warning, accuracy has decreased! Continuing search in case of local maxima.)")

        # print(f"OVERALL BEST SET: {[x+1 for x in overall_best_set]}")
        # print("On level", i+1, ", added feature", feature_to_add_at_this_level+1, "to current set")
        rounded_best_so_far = round(best_so_far_accuracy * 100, 5)
        print(f"Feature set {[x+1 for x in current_set_of_features]} was best, accuracy is {rounded_best_so_far}%\n")

    rounded_overall = round(overall_best_accuracy * 100, 5)
    print(f"Finished search! The best feature subset is {[x+1 for x in overall_best_set]}, which has an accuracy of {rounded_overall}%\n")

def backward_feature_search_demo(data, numColumns): # TODO 
    current_set_of_features = list(range(numColumns)) # initialize full set
    overall_best_accuracy = 0
    overall_best_set = current_set_of_features.copy()

    for i in range(numColumns):
        feature_to_remove_at_this_level = None
        best_so_far_accuracy = 0
        if i == 0:
            accuracy = leave_one_out_cross_validation(data, current_set_of_features, None)
            rounded_accuracy = round(accuracy * 100, 5)
            print(f"    Using feature(s) {[x+1 for x in current_set_of_features]}, accuracy is {rounded_accuracy}%")
            best_so_far_accuracy = accuracy
        else:
            for k in current_set_of_features:
                temp_features = current_set_of_features.copy()
                temp_features.remove(k)
                accuracy = leave_one_out_cross_validation(data, temp_features, None) # no features being added
                rounded_accuracy = round(accuracy * 100, 5)
                print(f"    Using feature(s) {[x+1 for x in temp_features]}, accuracy is {rounded_accuracy}%")
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_remove_at_this_level = k
        print()

        if feature_to_remove_at_this_level is not None:
            current_set_of_features.remove(feature_to_remove_at_this_level)

            if best_so_far_accuracy > overall_best_accuracy:
                overall_best_accuracy = best_so_far_accuracy
                overall_best_set = current_set_of_features.copy()
            else:
                print("(Warning, accuracy has decreased! Continuing search in case of local maxima.)")

        # print(f"OVERALL BEST SET: {[x+1 for x in overall_best_set]}")
        # print("On level", i+1, ", added feature", feature_to_add_at_this_level+1, "to current set")
        rounded_best_so_far = round(best_so_far_accuracy * 100, 5)
        print(f"Feature set {[x+1 for x in current_set_of_features]} was best, accuracy is {rounded_best_so_far}%\n")

    base_accuracy = leave_one_out_cross_validation(data, [], None)
    rounded_base = round(base_accuracy * 100, 5)
    print(f"    Using feature(s) [], accuracy is {rounded_base}%\n")

    rounded_overall = round(overall_best_accuracy * 100, 5)
    print(f"Finished search! The best feature subset is {[x+1 for x in overall_best_set]}, which has an accuracy of {rounded_overall}%\n")


def main():
    print("Welcome to Dylan Vuong's Feature Selection Algorithm.")
    inputFile = input("Type in the name of the file to test: ")
    # inputFile = "CS170_Small_Data__20.txt"
    # inputFile = "CS170_Large_Data__104.txt"
    data = np.loadtxt(inputFile)
    numRows = data.shape[0]
    numColumns = data.shape[1]
    
    print("\nType the number of the algorithm you want to run.")

    menuOn = True
    while (menuOn):
        start_time = time.time()
        choice = int(input("\n1) Forward Selection\n2) Backward Elimination\n"))
        if choice == 1:
            print(f"\nThis dataset has {numColumns-1} features (not including the class attribute), with {numRows} instances.\n")
            start_time = time.time()
            forward_feature_search_demo(data, numColumns-1)
            end_time = time.time()
            print(f"Running time: {(end_time - start_time):.1f} seconds")
            break
        elif choice == 2:
            print(f"\nThis dataset has {numColumns-1} features (not including the class attribute), with {numRows} instances.\n")
            start_time = time.time()
            backward_feature_search_demo(data, numColumns-1)
            end_time = time.time()
            print(f"Running time: {(end_time - start_time):.1f} seconds")
            break
        else:
            print("Invalid choice.")
            continue
        
main()
