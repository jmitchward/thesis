


# Calculates the probability that a given sample belongs to a class
import basic_func


def class_probability(row, aboveFiveProb, belowFiveProb, summaries):
    aboveProb = aboveFiveProb
    belowProb = belowFiveProb
    for eachValue in range(41):
        aboveProb *= basic_func.probability(row[eachValue], summaries[eachValue][0], summaries[eachValue][1])
        belowProb *= basic_func.probability(row[eachValue], summaries[eachValue][0], summaries[eachValue][1])
    return aboveProb, belowProb


# Determines which class a given sample belongs to
def predict(row, summaries):
    # Returns the probability of each classification
    classProb = (basic_func.encoded_training_data[41].value_counts() / len(basic_func.encoded_training_data))
    # Probability of a sample belonging to 50000+
    aboveFiveProb = classProb[0]
    # Probability of a sample belonging to -50000
    belowFiveProb = classProb[1]
    probabilities = class_probability(row, aboveFiveProb, belowFiveProb, summaries)
    if probabilities[0] > probabilities[1]:
        return 0
    else:
        return 1


def main(data, testData):
    predictions = list()
    dataSize = len(testData)
    summaries = basic_func.basic_calc(data)
    for i in range(dataSize):
        row = testData.iloc[i]
        output = predict(row, summaries)
        predictions.append(output)
    print('Determining accuracy.')
    basic_func.accuracy(predictions, testData)

main(basic_func.encoded_training_data, basic_func.encoded_test_data)
