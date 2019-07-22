# Jason Ward 2017-2019

from math import exp
import pickle
import ML_base


# Its iloc[row][column]
# Features are coluns, datapoints are rows


class logistic_regression():

    def __init__(self, train_data, test_data, train_class, test_class):
        print("Training data retrieved.")
        # self.weight_key = 0.0
        # Initial weight per formula standard, then weights for each feature of the set
        self.weights = [0.0] * 41
        # Dataset to train the algorithm
        print("Test data retrieved.")
        # Dataset to score the weights
        self.test_data = test_data
        # Train set classifiers, separated for posterity
        self.train_class = train_class
        # Test set length will be different as well as classifiers
        self.test_class = test_class
        # Initial Declaration
        self.data = train_data
        self.classifier = train_class
        # This would be an excellent point to make sure the data is encoded
        self.main()

    def weight_calculator(self, learn, iterations):
        # Gradient descent is only used to establish weights, which are only established using
        # training data, therefore it will only ever need receive training data.
        for eachIter in range(iterations):
            sumError = 0
            for datapoints in range(len(self.data)):
                # There are 40 weights, one for each individual column
                # Each weight is built from the sum of the columns
                print("Updating Weights {:3.2%}".format(datapoints / (len(self.data))), end="\r")
                datapoint = self.data.iloc[datapoints]
                result = self.predictor(datapoint)
                error = (result - self.classifier[datapoints])
                sumError += .5 * (error ** 2)
                self.weights[0] = self.weights[0] - learn * (1 / (len(self.data))) * error * result
                # For each feature, use the LR algorithm to train the weight
                for i in range(len(self.data.columns) - 1):
                    next_value = self.data.iloc[datapoints][i]
                    self.weights[i + 1] = self.weights[i + 1] - learn * (1 / (len(self.data))) * error * next_value
        return self.weights

    def predictor(self, data_index):
        # Store the initial weight
        weight_key = self.weights[0]
        for feature in range(len(self.data.columns)):
            # Using the logistic function, calculate the predicted output
            feature_value = data_index[feature]
            weight_key += (self.weights[feature] * feature_value)
            # Multiply the weight by the actual value of each row in the feature
        if weight_key < 0:
            return 1.0 - 1 / (1.0 + exp(self.weights[0]))
        else:
            return 1.0 / (1.0 + exp(-self.weights[0]))

    def main(self):
        learningRate = 0.2
        iterations = 10
        # Returns the list of weights
        print("Calculating weights!")
        self.weights = self.weight_calculator(learningRate, iterations)
        predictions = list()
        # Training phrase is complete, redefine working dataset as the test data
        self.data = self.test_data
        self.classifier = self.test_class
        for row in range(len(self.data)):
            print("Predicting {:3.2%}".format(row / (len(self.data))), end="\r")
            row = self.data.iloc[row]
            prediction = self.predictor(row)
            # Rounds prediction result to 2 decimal places.
            if prediction > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)

        self.results = ML_base.machine_learning.log_reg_accuracy(self.data, predictions, self.test_class)
        ML_base.machine_learning.save_instance(self)
