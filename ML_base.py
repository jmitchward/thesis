# Jason Ward 2017-2019

import pandas as pd
from math import sqrt
from math import pi
from math import exp
import pickle


# NOTE: Is it faster to put testing data and training data in two seperate class instances to eliminate passing
# the dataset being used in a function every time? The weights for LR would have to be passed between after the test
# data is completed, the NB percentages the same. How many functions would I need to rewrite to accommodate? Later.

# NOTE: Should these be static? Does their function depend on the class instance?

# Import the data from the UCI repository, assign it to separate variables
class machine_learning:
    # 41 total features of the data set, 31 require encoding
    numFeatures = []
    catFeatures = []
    questionFeatures = []

    train_data = pd.read_csv('./ml_data/census_income_real.data', header=None)
    test_data = pd.read_csv('./ml_data/census_income_test.test', header=None)

    #   train_data = pd.read_csv(
    #    'http://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz', header=None)
    #   test_data = pd.read_csv(
    #   'https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz',
    #    header=None)

    #    def __init__(self):
    #        # Request dataframes when class is initialized for future use
    #        data_pick = input("Which dataset is being used?")
    #        if data_pick.lower() == "test":
    #            print("Formatting testing data.")
    #            self.data = self.test_data
    #            self.format_data()
    #        elif data_pick.lower() == "train":
    #            print("Formatting training data.")
    #            self.data = self.train_data
    #            self.format_data()
    #
    #        else:
    #            print("Quit fucking around.")
    #            self.data = self.train_data
    #            self.format_data()

    @staticmethod
    def encode_values(data, column):
        data.loc[:, column] = data.loc[:, column].astype('category')
        data.loc[:, column] = data.loc[:, column].cat.codes
        data.loc[:, column] = data.loc[:, column].astype('int')

    @staticmethod
    def format_data(data, feature_set):

        machine_learning.catFeatures = feature_set[0]
        machine_learning.numFeatures = feature_set[1]

        # Abstract function definition being avoided for testing purposes
        print('Encoding categorical features.')
        for value in machine_learning.catFeatures:
            data[value].replace(' ?', data.describe(include='all')[value][2], inplace=True)
        for each in machine_learning.catFeatures:
            machine_learning.encode_values(data, each)
        # Take the categorical features, feed them to the encode function.
        print('Encoding numerical features.')
        for each in machine_learning.numFeatures:
            mean, std = data[each].mean(), data[each].std()
            data.iloc[:, each] = (data[each] - mean) / std
        # Standardize categorical features
        print('Standardizing categorical features.')
        for each in machine_learning.catFeatures:
            data.loc[:, each] = (data[each] - data[each].mean()) / data[each].std()
        return data

    # Calculates probability
    @staticmethod
    def probability(value, mean, sdev):
        condProb = exp(-((value - mean) ** 2 / (2 * sdev ** 2)))
        return (1 / (sqrt(2 * pi) * sdev)) * condProb

    # Calculates the mean and standard deviation for each column
    @staticmethod
    def basic_calc(data):
        summaries = list()

        # for each in hiImpact:
        for each in len(data.columns):
            mean = data[each].mean()
            sdev = data[each].std()
            # creates a list containing the mean and s-dev for each feature in the set.
            summaries.append([mean, sdev])
        return summaries

    @staticmethod
    def log_reg_accuracy(data, predict, classifier):
        correct = 0
        print('')
        dSize = len(data)
        for i in range(dSize):
            print("Scoring {:3.2%}".format(i / (len(data))), end="\r")
            if classifier[i] == predict[i]:
                correct += 1
        print('')
        print('Accuracy:', round(((correct / dSize) * 100.0)), '%')

    @staticmethod
    def accuracy(data, predict):
        # This function does not work without the test data formatted. FIX!
        print('Beginning accuracy rating.')
        correct = 0
        dSize = len(data)
        for i in range(dSize):
            if data.iloc[i][41] == predict[i]:
                correct += 1
        print('Accuracy:', round(((correct / dSize) * 100.0)), '%')

    def save_instance(self):
        with open('./ml_data/lr_output', 'wb') as log_output:
            pickle.dump(self, log_output)

    def load_instance(self):
        with open('./ml_data/lr_output', 'rb') as load_file:
            load_instance = pickle.load(load_file)
            return load_instance
