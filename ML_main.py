# Do import calls as needed
# To be used for the ML foundation
import pandas as pd

import ML_LR
import ML_NB
import ML_DT
import ML_base
import dataframe_sort

print("Welcome. The default dataset is loaded. ")


# It makes more sense for the menu to inherit the machine learning class than anything else

class menu:

    def __init__(self):
        self.return_list = []
        self.train_class = []
        self.test_class = []

        self.train_data = pd.read_csv('./ml_data/census_income_real.data', header=None)
        # http://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz
        self.test_data = pd.read_csv('./ml_data/census_income_test.test', header=None)
        # https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz

        self.train_data = self.train_data[:100]
        self.test_data = self.test_data[:100]

        self.menu()

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

        self.menu()

    def format_dataset(self):
        choice_format = input("Which dataset would you like to format?")

        if choice_format.lower() == "train data":
            print("Formatting train data")
            self.return_list = self.format_chain(self.train_data)
            self.train_data = self.return_list[0]
            self.train_class = self.return_list[1]

        elif choice_format.lower() == "test data":
            print("Formatting test data...")
            self.return_list = self.format_chain(self.test_data)
            self.test_data = self.return_list[0]
            self.test_class = self.return_list[1]

        elif choice_format.lower() == "both":
            print("Formatting both...")
            print("Training Data:")
            self.return_list = self.format_chain(self.train_data)
            self.train_data = self.return_list[0]
            self.train_class = self.return_list[1]

            print("Test Data:")
            self.return_list = self.format_chain(self.test_data)
            self.test_data = self.return_list[0]
            self.test_class = self.return_list[1]

        self.menu()

    def run_ml_lr(self):
        print("Beginning Logistic Regression")
        self.log_reg = ML_LR.logistic_regression(self.train_data, self.test_data, self.train_class, self.test_class)
        self.menu()

    def run_ml_dt(self):
        print("Beginning Decision Tree")
        self.d_tree = ML_DT.decision_tree.main(self.train_data, self.test_data, self.train_class, self.test_class)
        self.menu()

    def run_ml_nb(self):
        print("Beginning Naive Bayes")
        self.n_bayes = ML_NB.naive_bayes(self.train_data, self.test_data, self.train_class, self.test_class)
        self.menu()

    # def run_predictions(self):
    #  choice_predict = input("Individual or group prediction?")
    #        if choice_predict == "individual":
    # Iterate over the dataset features asking for the values
    # of this singular entry
    # THEN
    #            log_reg.predict()
    # This is calling something that was not defined with this is chosen first
    # While respecting the parameter of the function call
    #        elif choice_predict == "group":
    # Request either a pre defined new list of entries or manual input

    # In order to predict for any new input, there needs to be established
    # a dictionary correlating the encoded values to the actual categorical
    # feature.

    # That is best done while encoding is happening

    def menu(self):
        print("1. Import Data")
        print("2. Format Data")
        print("3. Name Features")
        print("4. Run Logistic Regression")
        print("5. Run Naive Bayes")
        print("6. Run Decision Tree")
        print("7. Run Predictions ")
        print("8. Exit")

        next_choice = input("What would you like to do?")

        self.menu_select(next_choice)

    def menu_select(self, choice):
        if choice.lower == "import data" or str(choice) == "1":
            self.import_dataset()
        elif choice.lower() == "format data" or str(choice) == "2":
            self.format_dataset()
        elif choice.lower() == "name features" or str(choice) == "3":
            ML_base.machine_learning.feature_define(self.train_data)
        elif choice.lower() == "run logistic regression" or str(choice) == "4":
            self.run_ml_lr()
        elif choice.lower() == "run naive bayes" or str(choice) == "5":
            self.run_ml_nb()
        elif choice.lower() == "run decision tree" or str(choice) == "6":
            self.run_ml_dt()
        # elif choice.lower() == "run predictions" or str(choice) == "7":
        #    self.run_predictions()
        elif choice.lower() == "exit" or str(choice) == "8":
            exit()
        else:
            print("Invalid selection.")
            self.menu()


menu()
