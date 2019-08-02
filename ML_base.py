# Jason Ward 2017-2019

from math import sqrt
from math import pi
from math import exp
import pickle


class machine_learning:
    numFeatures = []
    catFeatures = []
    questionFeatures = []

    @staticmethod
    def encode_values(data, column):
        data[column] = data[column].astype('category')
        data[column] = data[column].cat.codes
        data[column] = data[column].astype('int')

    @staticmethod
    def format_data(data, feature_set):

        machine_learning.catFeatures = feature_set[0]
        machine_learning.numFeatures = feature_set[1]

        print("Searching for illegal characters.")
        for value in machine_learning.catFeatures:
            data[value].replace(' ?', data.describe(include='all')[value][2], inplace=True)

        print('Encoding categorical features.')
        for each in machine_learning.catFeatures:
            machine_learning.encode_values(data, each)

        print('Encoding numerical features.')
        for each in machine_learning.numFeatures:
            mean, std = data[each].mean(), data[each].std()
            data.iloc[:, each] = (data[each] - mean) / std

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
        for each in range(len(data.columns)):
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
        return round(((correct / dSize) * 100.0))

    @staticmethod
    def accuracy(classifier, predict):
        print('Beginning accuracy rating.')
        correct = 0
        for i in range(len(classifier)):
            print("Scoring {:3.2%}".format(i / (len(classifier))), end="\r")
            if classifier[i] == predict[i]:
                correct += 1
        print('Accuracy:', round(((correct / len(classifier)) * 100.0)), '%')

    @staticmethod
    def feature_define(data):
        print("Please enter the feature names.")
        feature_names = []

        for each in range(len(data.columns)):
            print(data.iloc[:5][each])
            feature_title = input("What is the title of this feature?")
            feature_names.append(feature_title)
        return feature_names

    def save_instance(self):
        with open('./ml_data/lr_output', 'wb') as log_output:
            pickle.dump(self, log_output)

    def load_instance(self):
        with open('./ml_data/lr_output', 'rb') as load_file:
            load_instance = pickle.load(load_file)
            return load_instance
