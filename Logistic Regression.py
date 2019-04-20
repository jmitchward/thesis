from math import exp

import basic_func


def gradient_descent(data, learn, iterations):
    # Returns list of 41 zeros
    weights = [0.0] * 41
    # For the pre-determined number of iterations
    for eachIter in range(iterations):
        sumError = 0
        for row in range(len(data)):
            print("Updating Weights {:3.2%}".format(row / (len(data))), end="\r")
            row = data.iloc[row]
            result = predict(row, weights)
            error = (result - row[41])
            sumError += (.5) * (error ** 2)
            weights[0] = weights[0] - learn * (1 / (len(data))) * error * result
            for i in range(40):
                weights[i + 1] = weights[i + 1] - learn * (1 / (len(data))) * error * row[i]
    return weights


def predict(row, weights):
    # Store the initial weight
    weight = weights[0]
    for each in range(40):
        # Using the logistic function, calculate the predicted output
        weight += (weights[each + 1] * row[each])
    if weight < 0:
        return 1.0 - 1 / (1.0 + exp(weight))
    else:
        return 1.0 / (1.0 + exp(-weight))


def main(data, testData):
    learningRate = 0.2
    iterations = 10
    # Returns the list of weights
    weights = gradient_descent(data, learningRate, iterations)
    predictions = list()
    for row in range(len(testData)):
        print("Predicting {:3.2%}".format(row / (len(testData))), end="\r")
        row = testData.iloc[row]
        prediction = predict(row, weights)
        # Rounds prediction result to 2 decimal places.
        if prediction > 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
    basic_func.log_reg_accuracy(predictions, testData)


trainSet = basic_func.encoded_training_data
testSet = basic_func.encoded_test_data
# Limits the dataset to the first 50k entries. Used to save time.

print('Beginning Logistic Regression algorithm.')
main(trainSet, testSet)
