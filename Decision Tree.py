import basic_func


def class_count(branch):
    class0, class1 = 0, 0
    for eachRow in range(len(branch)):
        if branch[eachRow][41] == 1.0:
            class0 += 1
        else:
            class1 += 1
    return class0, class1


def first_test(index, value, dataset):
    leftBranch = list()
    rightBranch = list()
    for row in range(len(dataset)):
        print("Progress {:3.2%}".format(row / (len(dataset))), end="\r")
        # If the value stored at the index value for a given row/feature is less than or greater than target value
        if dataset.iloc[row][index] < value:
            # take the sample and add it to the left list, indicating a split upon that chosen feature value
            leftBranch.append(dataset.iloc[row])
        else:
            rightBranch.append(dataset.iloc[row])
    # print('Returning test split.')
    return leftBranch, rightBranch


def branch_test(index, value, dataset):
    leftBranch = list()
    rightBranch = list()
    for row in range(len(dataset)):
        print("Progress {:3.2%}".format(row / (len(dataset))), end="\r")
        # If the value stored at the index value for a given row/feature is less than or greater than target value
        if dataset[row][index] < value:
            # take the sample and add it to the left list, indicating a split upon that chosen feature value
            leftBranch.append(dataset[row])
        else:
            rightBranch.append(dataset[row])
    # print('Returning test split.')
    return leftBranch, rightBranch


def score(branches):
    includedSamples = float(len(branches[0]) + len(branches[1]))
    gini = 0.0
    for branch in branches:
        size = float(len(branch))
        score = 0.0
        if size == 0:
            continue
        class0, class1 = class_count(branch)
        # used for gini index, sum of the proportions squared
        proportion0 = (class0 / size)
        proportion1 = (class1 / size)
        score = (proportion0 ** 2) + (proportion1 ** 2)
        # Gini index = 1 - sum(proportions.squared) *
        gini += (1.0 - score) * (size / includedSamples)
    # print('Returning Gini Index.')
    return gini


def first_branch(data):
    # The feature used for a branch
    branchIndex = 100
    # The value used to create branches
    branchValue = 100
    # The gini score of a decision
    branchScore = 100
    # The two branches created
    branchFound = None
    for index in hi_impact:
        print('Processing feature: ', index, end="\r")
        print('')
        for row in range(len(data)):
            value = data.iloc[row][index]
            branches = first_test(index, value, data)
            print("Progress {:3.2%}".format(row / (len(data))), end="\r")
            gini = score(branches)
            if gini < branchScore:
                branchIndex = index
                branchValue = data.iloc[row][index]
                branchScore = gini
                branchFound = branches
    # returns a dict entry detailing results needed for split
    print('Branching on feature ', branchIndex)
    return {'index': branchIndex, 'value': branchValue, 'branches': branchFound}


def next_branch(data):
    # The feature used for a branch
    branchIndex = 100
    # The value used to create branches
    branchValue = 100
    # The gini score of a decision
    branchScore = 100
    # The two branches created
    branchFound = None
    for index in hi_impact:
        for row in range(len(data)):
            print("Calculating Gini Scores {:3.2%}".format(row / len(data)), end="\r")
            value = data[row][index]
            branches = branch_test(index, value, data)
            gini = score(branches)
            if gini < branchScore:
                branchIndex = index
                branchValue = data[row][index]
                branchScore = gini
                branchFound = branches
    # returns a dict entry detailing results needed for split
    print('Branching on feature ', branchIndex)
    return {'index': branchIndex, 'value': branchValue, 'branches': branchFound}


def leaf(branch):
    # Take the class value from each sample remaining in the branch
    print('Making leaf!')
    counts = [branch[eachRow][41] for eachRow in range(len(branch))]
    return max(set(counts), key=counts.count)


def branch_split(branch, maxDepth, minSize, depth):
    print('Now with ', depth, ' branches!')
    includedSamples = float(len(branch['branches'][0]) + len(branch['branches'][1]))
    print('Partition size', includedSamples)
    # assign the decided branches for splitting
    branch['leftBranch'] = branch['branches'][0]
    branch['rightBranch'] = branch['branches'][1]
    leftBranch, rightBranch = branch['branches']
    # remove the branches from the remaining set to be split
    del (branch['branches'])
    # If no split was made, end splitting
    if not leftBranch:
        branch['leftBranch'] = branch['rightBranch'] = leaf(leftBranch + rightBranch)
        return
    if not rightBranch:
        branch['leftBranch'] = branch['rightBranch'] = leaf(leftBranch + rightBranch)
        return
    # check if the depth reached is beyond the max depth set
    if depth >= maxDepth:
        # If it has, terminate the branch and store it in the branch dict
        branch['leftBranch'] = leaf(leftBranch)
        branch['rightBranch'] = leaf(rightBranch)
        return
    # if the left branch holds fewer samples than the min size, terminate the branch
    if len(leftBranch) <= minSize:
        branch['leftBranch'] = leaf(leftBranch)
    # otherwise, find the next split and send it to this function to continue the process
    else:
        branch['leftBranch'] = next_branch(leftBranch)
        # Call self, increase depth as a new branch will be made
        branch_split(branch['leftBranch'], maxDepth, minSize, depth + 1)
    # if the right branch holds fewer samples than the min size, terminate the branch
    if len(rightBranch) <= minSize:
        branch['rightBranch'] = leaf(rightBranch)
    # otherwise, find the next split and send it to this function to continue the process
    else:
        branch['rightBranch'] = next_branch(rightBranch)
        # Call self, increase depth as a new branch will be made
        branch_split(branch['rightBranch'], maxDepth, minSize, depth + 1)
    return branch


def start(data, maxDepth, minSize):
    groot = first_branch(data)
    print(groot['index'], 'says I am Groot!')
    depth = 0
    branch_split(groot, maxDepth, minSize, depth + 1)
    return groot


def predict(branch, row, data):
    # if the value stored is less than the decision value
    index = branch['index']
    value = branch['value']
    if data[index][row] < value:
        # and if the branch is a dict, meaning it has more features than one
        if isinstance(branch['leftBranch'], dict):
            return predict(branch['leftBranch'], row, data)
        else:
            return branch['leftBranch']
    else:
        if isinstance(branch['rightBranch'], dict):
            return predict(branch['rightBranch'], row, data)
        else:
            return branch['rightBranch']


def main(data, testData):
    tree = start(data, 6, 50)
    # Begins tree with dataset providing the max number of branches
    print('Beginning predictions')
    predictions = list()
    for row in range(len(testData)):
        prediction = predict(tree, row, testData)
        predictions.append(prediction)
    print('Determining accuracy.')
    basic_func.accuracy(predictions, testData)
    return tree


limited_dataset = basic_func.encoded_training_data[:200]

non_impact = [11, 13, 6, 10, 5, 25, 14, 26, 27, 29, 21, 33, 37, 28, 20, 32, 40, 24, 34, 35]
hi_impact = [3, 39, 9, 30, 19, 23, 1, 2, 8, 0, 22, 18, 7, 16, 31, 38, 12, 15, 4, 17, 36]

# Format decision tree variables to facilitate faster learning process.
decision_tree_train = limited_dataset.drop(non_impact, axis=1, inplace=True)

print('Beginning Decision Tree algorithm.')
main(limited_dataset, basic_func.encoded_test_data)
