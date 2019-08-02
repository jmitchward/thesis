def dataset_discovery(data):
    # List for each of the feature types
    categorical = []
    # Number of features in the dataset

    print("Welcome to Data Discovery!")

    data_type = input("Will this be supervised?")
    if data_type.lower() == "yes":
        classifier = input("What column will the classifier be found?")
        classifier = int(classifier)

    #    elif data_type.lower() == "unsupervised":
    #        classifier = 999
    #    else:
    #        print("Invalid selection.")
    #        return

    for every in range(len(data.columns)):
        for each in range(len(data)):
            # for each column, use every row up to 100
            if type(data.iloc[every][each]) == str:
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

#    print("Discovered", len(categorical), "categorical features.")
#    for feature in range(len(categorical)):
#        print(categorical[feature], end=",")
#
#    print("\nDiscovered", len(numerical), " numerical features.")
#    for features in range(len(numerical)):
#        print(numerical[features], end=",")

#    doubleCheck = input("Is this correct?")

#    if doubleCheck.lower() == "yes":
    return categorical, numerical, classifier
#    else:
#        dataset_discovery(data)
