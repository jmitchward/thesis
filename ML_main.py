# Do import calls as needed
# To be used for the ML foundation
import pandas as pd

import ML_LR
import ML_NB
import ML_base
import dataframe_sort

print("Welcome. The default dataset is loaded. ")


def menu():
    global train_class
    global test_class
    exit_choice = 0
    feature_set = []
    feature_set_test = []
    # Pure for testing purposes.
    train_data = pd.read_csv('./ml_data/census_income_real.data', header=None)
    # http://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz
    test_data = pd.read_csv('./ml_data/census_income_test.test', header=None)
    # https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz

    train_data = train_data[:200]
    test_data = test_data[:200]

    print("1. Import database")
    print("2. Format database")
    print("3. Run Logistic Regression")
    print("4. Run Naive Bayes")
    print("5. Run Decision Tree")
    print("6. Run Predictions ")
    print("7. Exit")

    choice = input("What would you like to do?")

    while exit_choice == 0:

        if choice.lower() == 'import':
            # Input control please
            TTD = input("Would you like to import the training and test data separately or split automatically?")
            if TTD == "separately":
                url = input("Enter the URL for the training data:")
                train_data = pd.read_csv(url)
                feature_set = dataframe_sort.dataset_discovery(train_data)

                menu()
                url = input("Enter the URL for the test data:")
                test_data = pd.read_csv(url)
                feature_set = dataframe_sort.dataset_discovery(test_data)
                menu()
                # Successful and fail checks needed

        elif choice.lower() == 'format':
                        choice_format = input("Which dataset would you like to format?")
                        if choice_format.lower() == "train data":
                            feature_set = dataframe_sort.dataset_discovery(train_data)
                            print("Formatting training data.")
                            en_train = ML_base.machine_learning.format_data(train_data, feature_set, feature_set_test)
                            menu()
                        elif choice_format.lower() == "test data":
                            feature_set_test = dataframe_sort.dataset_discovery(test_data)
                            print("Formatting test data.")
                            en_test = ML_base.machine_learning.format_data(test_data, feature_set_test)
                            menu()
                        elif choice_format.lower() == "both":
                            print("Formatting both")
                            print("Training Data:")
                            feature_set = dataframe_sort.dataset_discovery(train_data)
                            print("Test Data:")
                            feature_set_test = dataframe_sort.dataset_discovery(test_data)
                            # Special formatting for the classifier to ensure it stays 0 or 1
                            ML_base.machine_learning.encode_values(train_data, feature_set[2])
                            ML_base.machine_learning.encode_values(test_data, feature_set_test[2])
                            # Copy classifier to a seperate list for safe keeping
                            train_class = train_data.iloc[:][feature_set[2]]
                            test_class = test_data.iloc[:][feature_set_test[2]]
                            # Remove classifier from the data to ensure accurate predictions
                            train_data = train_data.drop([feature_set[2]], axis=1)
                            test_data = test_data.drop([feature_set_test[2]], axis=1)
                            # In this instance, the datasets would use the same feature set
                            train_data = ML_base.machine_learning.format_data(train_data, feature_set)
                            test_data = ML_base.machine_learning.format_data(test_data, feature_set)
                            log_reg = ML_LR.logistic_regression(train_data, test_data, train_class, test_class)

                            menu()
                        else:
                            print("Invalid choice.")
                            menu()

        elif choice.lower() == 'logistic regression':

            print("Beginning Algorithm")
            log_reg = ML_LR.logistic_regression(train_data, test_data, train_class, test_class)
            print("Logistic Regression results saved.")
            # log_reg.save_instance()
            menu()
        #    elif choice == 4 or "decision tree":
        #        d_tree = ML_DR.decision_tree.main(train_data, test_data)

        elif choice.lower() == 'naive bayes':
            n_bayes = ML_NB.naive_bayes.main(e_trD, e_teD, feature_set)

        #    elif choice == 6 or "predict":
        #        choice_predict = input("Individual or group prediction?")
        #        if choice_predict == "individual":
        # Iterate over the dataset features asking for the values
        # of this singular entry
        # THEN
        #            log_reg.predict()
        # This is calling something that was not defined with this is chosen first
        # While respecting the parameter of the function call
        #        elif choice_predict == "group":
        # Request either a pre defined new list of entries or manual input

        elif choice.lower() == 'exit':
            exit_choice = 1
        else:
            menu()


menu()
