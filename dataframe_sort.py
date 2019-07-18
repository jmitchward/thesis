# isinstance(variable, type)
# isinstance(stringTest, str)

# This will be used to make the dataset ready to be
# processed.

import ML_base


def dataset_discovery(data):
    # List for each of the feature types
    categorical = []
    # Number of features in the dataset

    print("Welcome to Data Discovery!")

    data_type = input("Will this be supervised?")
    if data_type.lower() == "yes":
        classifier = input("What column will the classifier be found?")
        classifier = int(classifier)

        #ML_base.machine_learning.encode_values(data, classifier)

    #    elif data_type.lower() == "unsupervised":
    #        classifier = 999
    #    else:
    #        print("Invalid selection.")
    #        return

    for every in range(len(data.columns)):
        for each in range(len(data)):
            # for each column, use every row up to 100
            if isinstance(data[every][each], str):
                # If any value within that column is a string, it categorical
                categorical.append(every)
                # Add it to the list then break to the next column
                break
                # If it is a not a string, then it is a number

    # Make a list of the remaining, non-categorical features
    numerical = list(set(data.columns) - set(categorical))

    for eachFeature in categorical:
        if eachFeature == classifier:
            categorical.remove(classifier)
    for everyFeature in numerical:
        if everyFeature == classifier:
            numerical.remove(classifier)

    return categorical, numerical, classifier
