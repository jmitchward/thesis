# Jason Ward 2017-2019

import ML_base


class decision_tree:

    def __init__(self, train_data, test_data):
        print("Training data retrieved.")
        self.dataset = train_data
        print("Testing data retrieved.")
        self.test_data = test_data
        self.branch_class = []

        limited_dataset = self.dataset[:200]
        non_impact = [11, 13, 6, 10, 5, 25, 14, 26, 27, 29, 21, 33, 37, 28, 20, 32, 40, 24, 34, 35]
        hi_impact = [3, 39, 9, 30, 19, 23, 1, 2, 8, 0, 22, 18, 7, 16, 31, 38, 12, 15, 4, 17, 36]
        # Format decision tree variables to facilitate faster learning process.
        decision_tree_train = limited_dataset.drop(non_impact, axis=1, inplace=True)

    def class_count(self, branch):
        class0, class1 = 0, 0
        for eachRow in range(len(branch)):
            if branch[eachRow][41] == 1.0:
                self.branch_class[0] += 1
            else:
                self.branch_class[1] += 1
        return class0, class1

    def first_test(self, index, value):
        leftBranch = list()
        rightBranch = list()
        for row in range(len(self.dataset)):
            print("Progress {:3.2%}".format(row / (len(self.dataset))), end="\r")
            # If the value stored at the index value for a given row/feature is less than or greater than target value
            if self.dataset.iloc[row][index] < value:
                # take the sample and add it to the left list, indicating a split upon that chosen feature value
                leftBranch.append(self.dataset.iloc[row])
            else:
                rightBranch.append(self.dataset.iloc[row])
        # print('Returning test split.')
        return leftBranch, rightBranch

    def branch_test(self, index, value):
        leftBranch = list()
        rightBranch = list()
        for row in range(len(self.dataset)):
            print("Progress {:3.2%}".format(row / (len(self.dataset))), end="\r")
            # If the value stored at the index value for a given row/feature is less than or greater than target value
            if self.dataset[row][index] < value:
                # take the sample and add it to the left list, indicating a split upon that chosen feature value
                leftBranch.append(self.dataset[row])
            else:
                rightBranch.append(self.dataset[row])
        # print('Returning test split.')
        return leftBranch, rightBranch

    def score(self, branches):
        includedSamples = float(len(branches[0]) + len(branches[1]))
        gini = 0.0
        for branch in branches:
            size = float(len(branch))
            score = 0.0
            if size == 0:
                continue
            class0, class1 = self.class_count(branch)
            # used for gini index, sum of the proportions squared
            proportion0 = (class0 / size)
            proportion1 = (class1 / size)
            score = (proportion0 ** 2) + (proportion1 ** 2)
            # Gini index = 1 - sum(proportions.squared) *
            gini += (1.0 - score) * (size / includedSamples)
        # print('Returning Gini Index.')
        return gini

    def first_branch(self):
        # The feature used for a branch
        branchIndex = 100
        # The value used to create branches
        branchValue = 100
        # The gini score of a decision
        branchScore = 100
        # The two branches created
        branchFound = None
        for index in self.hi_impact:
            print('Processing feature: ', index, end="\r")
            print('')
            for row in range(len(self.data)):
                value = self.data.iloc[row][index]
                branches = self.first_test(index, value, self.data)
                print("Progress {:3.2%}".format(row / (len(self.data))), end="\r")
                gini = self.score(branches)
                if gini < branchScore:
                    branchIndex = index
                    branchValue = self.data.iloc[row][index]
                    branchScore = gini
                    branchFound = branches
        # returns a dict entry detailing results needed for split
        print('Branching on feature ', branchIndex)
        return {'index': branchIndex, 'value': branchValue, 'branches': branchFound}

    def next_branch(self, data):
        # The feature used for a branch
        branchIndex = 100
        # The value used to create branches
        branchValue = 100
        # The gini score of a decision
        branchScore = 100
        # The two branches created
        branchFound = None
        for index in self.hi_impact:
            for row in range(len(data)):
                print("Calculating Gini Scores {:3.2%}".format(row / len(data)), end="\r")
                value = data[row][index]
                branches = self.branch_test(index, value, data)
                gini = self.score(branches)
                if gini < branchScore:
                    branchIndex = index
                    branchValue = data[row][index]
                    branchScore = gini
                    branchFound = branches
        # returns a dict entry detailing results needed for split
        print('Branching on feature ', branchIndex)
        return {'index': branchIndex, 'value': branchValue, 'branches': branchFound}

    def leaf(self, branch):
        # Take the class value from each sample remaining in the branch
        print('Making leaf!')
        counts = [branch[eachRow][41] for eachRow in range(len(branch))]
        return max(set(counts), key=counts.count)

    def branch_split(self, branch, maxDepth, minSize, depth):
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
            branch['leftBranch'] = branch['rightBranch'] = self.leaf(leftBranch + rightBranch)
            return
        if not rightBranch:
            branch['leftBranch'] = branch['rightBranch'] = self.leaf(leftBranch + rightBranch)
            return
        # check if the depth reached is beyond the max depth set
        if depth >= maxDepth:
            # If it has, terminate the branch and store it in the branch dict
            branch['leftBranch'] = self.leaf(leftBranch)
            branch['rightBranch'] = self.leaf(rightBranch)
            return
        # if the left branch holds fewer samples than the min size, terminate the branch
        if len(leftBranch) <= minSize:
            branch['leftBranch'] = self.leaf(leftBranch)
        # otherwise, find the next split and send it to this function to continue the process
        else:
            branch['leftBranch'] = self.next_branch(leftBranch)
            # Call self, increase depth as a new branch will be made
            self.branch_split(branch['leftBranch'], maxDepth, minSize, depth + 1)
        # if the right branch holds fewer samples than the min size, terminate the branch
        if len(rightBranch) <= minSize:
            branch['rightBranch'] = self.leaf(rightBranch)
        # otherwise, find the next split and send it to this function to continue the process
        else:
            branch['rightBranch'] = self.next_branch(rightBranch)
            # Call self, increase depth as a new branch will be made
            self.branch_split(branch['rightBranch'], maxDepth, minSize, depth + 1)
        return branch

    def start(self, data, maxDepth, minSize):
        groot = self.first_branch(data)
        print(groot['index'], 'says I am Groot!')
        depth = 0
        self.branch_split(groot, maxDepth, minSize, depth + 1)
        return groot

    def predict(self, branch, row, data):
        # if the value stored is less than the decision value
        index = branch['index']
        value = branch['value']
        if data[index][row] < value:
            # and if the branch is a dict, meaning it has more features than one
            if isinstance(branch['leftBranch'], dict):
                return self.predict(branch['leftBranch'], row, data)
            else:
                return branch['leftBranch']
        else:
            if isinstance(branch['rightBranch'], dict):
                return self.predict(branch['rightBranch'], row, data)
            else:
                return branch['rightBranch']

    def main(self, testData):
        tree = self.start(self.data, 6, 50)
        # Begins tree with dataset providing the max number of branches
        print('Beginning predictions')
        predictions = list()
        for row in range(len(testData)):
            prediction = self.predict(tree, row, testData)
            predictions.append(prediction)
        print('Determining accuracy.')
        decision_tree.accuracy(predictions, testData)
        return tree




