# Jason Ward 2017-2019
# This is all rather abstract, thankfully. There is a reference to column 41, line 27
# Introduce a column count to replace 41 on line 19
# Both those things should ideally happen at the data entry point
# or I guess here? That'd be easy, but all that info has to be gleaned at the beginning

import ML_base


class naive_bayes(ML_base.machine_learning):

    def __init__(self, train_data, test_data):
        print("Training data retrieved.")
        self.train_data = train_data
        print("Testing data retrieved.")
        self.test_data = test_data

    def class_probability(self, row, aboveFiveProb, belowFiveProb, summaries):
        aboveProb = aboveFiveProb
        belowProb = belowFiveProb
        for eachValue in range(41):
            aboveProb *= self.probability(row[eachValue], summaries[eachValue][0], summaries[eachValue][1])
            belowProb *= self.probability(row[eachValue], summaries[eachValue][0], summaries[eachValue][1])
        return aboveProb, belowProb

    # Determines which class a given sample belongs to
    def predict(self, row, summaries):
        # Returns the probability of each classification
        classProb = (self.train_data[41].value_counts() / len(self.train_data))
        # Probability of a sample belonging to 50000+
        aboveFiveProb = classProb[0]
        # Probability of a sample belonging to -50000
        belowFiveProb = classProb[1]
        probabilities = self.class_probability(row, aboveFiveProb, belowFiveProb, summaries)
        if probabilities[0] > probabilities[1]:
            return 0
        else:
            return 1

    def main(self):
        predictions = list()
        dataSize = len(self.testData)
        summaries = self.basic_calc(self.train_data)
        for i in range(dataSize):
            row = self.test_data.iloc[i]
            output = self.predict(row, summaries)
            predictions.append(output)
        print('Determining accuracy.')
        self.accuracy(predictions, self.test_data)
