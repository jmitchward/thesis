# Jason Ward 2017-2019
# This is all rather abstract, thankfully. There is a reference to column 41, line 27
# Introduce a column count to replace 41 on line 19
# Both those things should ideally happen at the data entry point
# or I guess here? That'd be easy, but all that info has to be gleaned at the beginning

import ML_base


class naive_bayes:

    def __init__(self, train_data, test_data, train_class, test_class):
        print("Training data retrieved.")
        self.data = train_data
        print("Testing data retrieved.")
        self.test_data = test_data
        self.train_class = train_class
        self.test_class = test_class
        self.classProb = [0, 0]
        self.above_count = 0
        self.below_count = 0

        self.main()

    # Determines which class a given sample belongs to
    def predict(self, row, summaries):
        # Returns the probability of each classification
        self.class_count()
        # Probability of a sample belonging to 50000+
        # classProb[0]
        # Probability of a sample belonging to -50000
        # classProb[1]
        self.class_probability(row, summaries)
        if self.classProb[0] > self.classProb[1]:
            return 0
        else:
            return 1

    def class_count(self):
        local_above = 0
        local_below = 0
        for each in range(len(self.test_class)):
            if self.test_class[each] == 0:
                local_above += 1
            else:
                local_below += 1
        self.classProb[0] = local_above / len(self.data)
        self.classProb[1] = local_below / len(self.data)

    def class_probability(self, row, summaries):
        for eachValue in range(self.data.columns):
            self.classProb[0] *= ML_base.machine_learning.probability(row[eachValue], summaries[eachValue][0],
                                                                      summaries[eachValue][1])
            self.classProb[1] *= ML_base.machine_learning.probability(row[eachValue], summaries[eachValue][0],
                                                                      summaries[eachValue][1])

    def main(self):
        predictions = list()
        print("Calculating feature summaries.")
        summaries = ML_base.machine_learning.basic_calc(self.data)
        print("Beginning predictions.")
        for i in range(len(self.test_data)):
            print("Predicting {:3.2%}".format(i / (len(self.test_data))), end="\r")
            output = self.predict(self.test_data.iloc[i], summaries)
            predictions.append(output)
        print('Determining accuracy.')
        ML_base.machine_learning.accuracy(self.test_class, predictions)
