import math

from DecisonTree import Leaf, Question, DecisionNode, class_counts
from utils import *

"""
Make the imports of python packages needed
"""


class ID3:
    def __init__(self, label_names: list, min_for_pruning=0, target_attribute='diagnosis'):
        self.label_names = label_names
        self.target_attribute = target_attribute
        self.tree_root = None
        self.used_features = set()
        self.min_for_pruning = min_for_pruning

    @staticmethod
    def entropy(rows: np.ndarray, labels: np.ndarray):
        """
        Calculate the entropy of a distribution for the classes probability values.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: entropy value.
        """
        # TODO:
        #  Calculate the entropy of the data as shown in the class.
        #  - You can use counts as a helper dictionary of label -> count, or implement something else.

        counts = class_counts(rows, labels)
        impurity = 0.0

        # ====== YOUR CODE: ======
        total_count = sum(counts.values())
        h_entopry = [(list(counts.values())[i]/total_count) * math.log2(list(counts.values())[i]/total_count) for i in range(len(counts))]
        impurity = -sum(h_entopry)
        # ========================

        return impurity

    def info_gain(self, left, left_labels, right, right_labels, current_uncertainty):
        """
        Calculate the information gain, as the uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        :param left: the left child rows.
        :param left_labels: the left child labels.
        :param right: the right child rows.
        :param right_labels: the right child labels.
        :param current_uncertainty: the current uncertainty of the current node
        :return: the info gain for splitting the current node into the two children left and right.
        """
        # TODO:
        #  - Calculate the entropy of the data of the left and the right child.
        #  - Calculate the info gain as shown in class.

        entropy_right = self.entropy(right, right_labels)
        entropy_left = self.entropy(left, left_labels)
        
        total_size = len(left) + len(right)

        sigma = ((len(right) / total_size) * entropy_right) + ((len(left) / total_size) * entropy_left)
        
        return current_uncertainty - sigma


    def partition(self, rows, labels, question: Question, current_uncertainty):
        """
        Partitions the rows by the question.
        :param rows: array of samples
        :param labels: rows data labels.
        :param question: an instance of the Question which we will use to partition the data.
        :param current_uncertainty: the current uncertainty of the current node
        :return: Tuple of (gain, true_rows, true_labels, false_rows, false_labels)
        """
        # TODO:
        #   - For each row in the dataset, check if it matches the question.
        #   - If so, add it to 'true rows', otherwise, add it to 'false rows'.
        #   - Calculate the info gain using the `info_gain` method.
        
        # ====== YOUR CODE: ======
        gain, true_rows, true_labels, false_rows, false_labels = None, [], [], [], []

        for row, label in zip(rows, labels):
            if question.match(row):
                true_rows.append(row)
                true_labels.append(label)
            else:
                false_rows.append(row)
                false_labels.append(label)

        # current_uncertainty = self.entropy(rows, labels)
        gain = self.info_gain(true_rows, true_labels, false_rows, false_labels, current_uncertainty)

        # assert len(rows) == len(labels), 'Rows size should be equal to labels size.'
        return gain, true_rows, true_labels, false_rows, false_labels
        # ========================

    def find_best_split(self, rows, labels):
        """
        Find the best question to ask by iterating over every feature / value and calculating the information gain.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: Tuple of (best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels)
        """
        # ====== YOUR CODE: ======
        
        # TODO:
        #   - For each feature of the dataset, build a proper question to partition the dataset using this feature.
        #   - find the best feature to split the data. (using the `partition` method)
        best_gain = - math.inf  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        best_false_rows, best_false_labels = None, None
        best_true_rows, best_true_labels = None, None
        current_uncertainty = self.entropy(rows, labels)

        for feature in range(len(rows[0])):
            column = [rows[j][feature] for j in range(len(rows))]
            labeled_column = np.c_[np.array(column), np.array(labels)]
            sorted_labeled_column = sorted(labeled_column, key=lambda x: x[0])
            for val_index in range(1,len(sorted_labeled_column)):
                q = Question(column=[float(x[0]) for x in sorted_labeled_column], column_idx=feature, value=float(sorted_labeled_column[val_index][0]))
                gain, true_rows, true_labels, false_rows, false_labels = self.partition(rows, labels, q, current_uncertainty)
                if gain > best_gain:
                    best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels = \
                                                            gain, q, true_rows, true_labels, false_rows, false_labels

        return best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels
        # ========================

    def build_tree(self, rows, labels):
        """
        Build the decision Tree in recursion.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: a Question node, This records the best feature / value to ask at this point, depending on the answer.
                or leaf if we have to prune this branch (in which cases ?)

        """
        # TODO:
        #   - Try partitioning the dataset using the feature that produces the highest gain.
        #   - Recursively build the true, false branches.
        #   - Build the Question node which contains the best question with true_branch, false_branch as children
        best_question = None
        true_branch, false_branch = None, None

        # ====== YOUR CODE: ======
        best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels = self.find_best_split(rows, labels)
        if not len(set(best_true_labels)) == 1:
            true_branch = self.build_tree(best_true_rows, best_true_labels)
        else: 
            return Leaf(best_true_rows, best_true_labels)
        if not len(set(best_false_labels)) == 1:
            false_branch = self.build_tree(best_false_rows, best_false_labels)
        else:
            return Leaf(best_false_rows, best_false_labels)

        return DecisionNode(best_question, true_branch, false_branch)

        # ========================


    def fit(self, x_train, y_train):
        """
        Trains the ID3 model. By building the tree.
        :param x_train: A labeled training data.
        :param y_train: training data labels.
        """
        # TODO: Build the tree that fits the input data and save the root to self.tree_root

        # ====== YOUR CODE: ======

        self.tree_root = self.build_tree(x_train, y_train)
        return

        # raise NotImplementedError
        # ========================

    def predict_sample(self, row, node: DecisionNode or Leaf = None):
        """
        Predict the most likely class for single sample in subtree of the given node.
        :param row: vector of shape (1,D).
        :return: The row prediction.
        """
        # TODO: Implement ID3 class prediction for set of data.
        #   - Decide whether to follow the true-branch or the false-branch.
        #   - Compare the feature / value stored in the node, to the example we're considering.
        
        if node is None:
            node = self.tree_root
        prediction = None

        # ====== YOUR CODE: ======

        while not isinstance(node, Leaf):
            next_branch = node.question.match(row)
            node = node.true_branch if next_branch else node.false_branch
        prediction = max(node.predictions, key=node.predictions.get)
            
        return prediction
        # ========================


    def predict(self, rows):
        """
        Predict the most likely class for each sample in a given vector.
        :param rows: vector of shape (N,D) where N is the number of samples.
        :return: A vector of shape (N,) containing the predicted classes.
        """
        # TODO:
        #  Implement ID3 class prediction for set of data.

        # y_pred = None

        # ====== YOUR CODE: ======
        y_pred = []
        for row in rows:
            prediction = self.predict_sample(row, self.tree_root)
            y_pred.append(prediction)
        return y_pred
        # ========================

