# Do import calls as needed
# To be used for the ML foundation
import pandas as pd

import ML_LR
import ML_NB
import ML_DT
import ML_base
import dataframe_sort

print("Welcome. The default dataset is loaded. ")


class menu:

    def __init__(self):
        self.exit_choice = 0
        self.ml_choice = 0
        self.feature_set = []
        self.feature_set_test = []
        self.train_data = pd.read_csv('./ml_data/census_income_real.data', header=None)
        # http://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz
        self.test_data = pd.read_csv('./ml_data/census_income_test.test', header=None)
        # https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz

        self.train_data = self.train_data[:200]
        self.test_data = self.test_data[:200]

        self.menu_choice()

    def format_chain(self, data):
        # Sort the dataset features into categorical and numerical
        features = dataframe_sort.dataset_discovery(data)
        # Encode binary classifier into 0 or 1
        ML_base.machine_learning.encode_values(data, features[2])
        # Separate classifier from dataset
        classifiers = data.iloc[:][features[2]]
        # Drop classifier from dataset
        data = data.drop([features[2]], axis=1)
        # Standardize dataset
        data = ML_base.machine_learning.format_data(data, features)

        return data, classifiers

    def import_dataset(self):
        TTD = input("Would you like to import the training and test data separately or split automatically?")

        if TTD == "separately":
            url = input("Enter the URL for the training data:")
            train_data = pd.read_csv(url)
            self.feature_set = dataframe_sort.dataset_discovery(train_data)

            url = input("Enter the URL for the test data:")
            test_data = pd.read_csv(url)
            self.feature_set_test = dataframe_sort.dataset_discovery(test_data)

    def format_dataset(self):
        choice_format = input("Which dataset would you like to format?")

        if choice_format.lower() == "train data":
            print("Formatting train data")
            self.train_data, self.train_class = self.format_chain(self.train_data)

        elif choice_format.lower() == "test data":
            print("Formatting test data...")
            self.test_data, self.test_class = self.format_chain(self.test_data)

        elif choice_format.lower() == "both":
            print("Formatting both...")
            print("Training Data:")
            self.train_data, self.train_class = self.format_chain(self.train_data)
            print("Test Data:")
            self.test_data, self.test_class = self.format_chain(self.test_data)

    def run_ml_lr(self):
        print("Beginning Logistic Regression")
        log_reg = ML_LR.logistic_regression(self.train_data, self.test_data, self.train_class, self.test_class)

    def run_ml_dt(self):
        print("Beginning Decision Tree")
        d_tree = ML_DT.decision_tree.main(self.train_data, self.test_data, self.train_class, self.test_class)

    def run_ml_nb(self):
        print("Beginning Naive Bayes")
        n_bayes = ML_NB.naive_bayes.main(self.train_data, self.test_data, self.train_class, self.test_class)

    def menu(self):
        print("1. Import Data")
        print("2. Format Data")
        print("3. Run Logistic Regression")
        print("2. Run Naive Bayes")
        print("3. Run Decision Tree")
        print("4. Run Predictions ")

        next_choice = input("What would you like to do?")

        return next_choice

    def menu_select(self, choice):
        

    def exit_choice(self):
        self.exit_choice = 1

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
