import numpy
from sklearn.impute import KNNImputer
from sklearn import metrics
from utils import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# The code for part B was inspired from the following link
# http://ethen8181.github.io/machine-learning/recsys/1_ALSWR.html


np.random.seed(9564)


class MatrixCompletion:
    def __init__(self, num_iter, rank):
        self.num_iter = num_iter
        self.rank = rank
        self.num_user = None
        self.num_question = None
        self.user_factors = None
        self.item_factors = None
        self.train_record = None

    def fit(self, train):
        self.num_user, self.num_question = train.shape
        self.user_factors = np.random.random((self.rank, self.num_user))
        self.item_factors = np.random.random((self.rank, self.num_question))

        self.train_record = []

        for _ in range(self.num_iter):
            self.user_factors = self.step(train.transpose(), self.user_factors, self.item_factors)
            self.item_factors = self.step(train, self.item_factors, self.user_factors)
            predictions = self.predict()
            train_error = self.compute_error(train, predictions)
            self.train_record.append(train_error)

        return self

    def step(self, train, unfixed_var, fixed_var):
        term_1 = fixed_var.dot(fixed_var.transpose())
        term_1_inverse = np.linalg.inv(term_1)
        term_2 = term_1_inverse.dot(fixed_var)
        final = term_2.dot(train)

        return final

    def predict(self):
        predictions = self.user_factors.transpose().dot(self.item_factors)
        return predictions

    def compute_error(self, train, predictions):
        mask = np.nonzero(train)
        error = mean_squared_error(train[mask], predictions[mask])
        return error


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def knn_impute_by_item_roc(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix.T)
    acc, true_labels, predict_labels = sparse_matrix_evaluate_roc(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc, true_labels, predict_labels


def plot_learning_curve(train_record, rank):
    fig = plt.figure()
    plt.figure().clear()
    linewidth = 3
    iterations = [i for i in range(1, 101)]
    plt.plot(iterations, train_record)
    plt.xlabel("Iterations")
    plt.ylabel("Mean Squared Error")
    plt.title("Training Error with Rank=" + str(rank))
    plt.savefig("Matrix_Completion with Rank=" + str(rank))


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    '''
    question_data = load_question_meta("../data")

    topics = question_data["topics"]
    print(topics[0].strip('][').split(', '))

    record = [0 for _ in range(389)]

    for topic in topics:
        topic_strip = topic.strip('][').split(', ')
        for subject in topic_strip:
            record[int(subject)] += 1

    indices = []

    for _ in range(5):
        index_max = record.index(max(record))
        indices.append(index_max)

        record[index_max] = -1

    print(indices)
    '''

    # Preprocess sparse matrix
    zero_train_matrix = sparse_matrix.copy()
    zero_train_matrix[np.isnan(sparse_matrix)] = 0
    zero_train_matrix[sparse_matrix == 0] = -1

    rank_values = [5, 10, 20, 50, 100, 200, 300, 400, 500]
    num_iter = 100

    matrixcomp_val_accuracies = []

    # Train the 9 models
    for rank in rank_values:
        model = MatrixCompletion(num_iter=num_iter, rank=rank)
        model.fit(zero_train_matrix)
        plot_learning_curve(model.train_record, rank)

        reconstructed_matrix = model.user_factors.transpose().dot(model.item_factors)

        val_acc = sparse_matrix_evaluate(val_data, reconstructed_matrix, 0)

        matrixcomp_val_accuracies.append(val_acc)

        print("Model for rank", rank, "complete")

    # Plot Validation accuracies
    fig = plt.figure()
    plt.figure().clear()
    plt.plot(rank_values, matrixcomp_val_accuracies)
    plt.xlabel("Rank")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs Rank Value")
    plt.savefig("Validation vs Rank")

    # Find the optimal value for rank
    index_max = matrixcomp_val_accuracies.index(max(matrixcomp_val_accuracies))
    rank_opt = rank_values[index_max]

    print("Rank Optimal:", rank_opt)

    # Get test accuracy
    model_opt = MatrixCompletion(num_iter=100, rank=rank_opt)
    model_opt.fit(zero_train_matrix)
    opt_reconstruct_matrix = model_opt.user_factors.transpose().dot(model_opt.item_factors)

    test_acc = sparse_matrix_evaluate(test_data, opt_reconstruct_matrix, 0)
    print("Test Accuracy with optimal rank", test_acc)

    '''
    #####################################################################
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_values = [1, 6, 11, 16, 21, 26]

    print("User-based:")
    accuracies = [knn_impute_by_user(sparse_matrix, val_data, k) for k in k_values]
    plt.plot(k_values, accuracies)
    plt.xlabel("k")
    plt.ylabel("Validation Accuracy")
    plt.savefig("knn_user.png")
    plt.show()

    index = accuracies.index(max(accuracies))
    k_opt = k_values[index]
    print(f"Test Accuracy with k*={k_opt}:", knn_impute_by_user(sparse_matrix, test_data, k_opt))
    print()

    print("Item-based:")
    accuracies = [knn_impute_by_item(sparse_matrix, val_data, k) for k in k_values]
    plt.plot(k_values, accuracies)
    plt.xlabel("k")
    plt.ylabel("Validation Accuracy")
    plt.savefig("knn_item.png")
    plt.show()

    index = accuracies.index(max(accuracies))
    k_opt = k_values[index]
    print(f"Test Accuracy with k*={k_opt}:", knn_impute_by_item(sparse_matrix, test_data, k_opt))

    accuracy, true_labels, predict_labels = knn_impute_by_item_roc(sparse_matrix, val_data, 21)
    fpr, tpr, _ = metrics.roc_curve(true_labels, predict_labels)
    auc = metrics.roc_auc_score(true_labels, predict_labels)

    plt.plot(fpr, tpr,label="AUC="+str(auc))
    plt.ylabel("True Positive")
    plt.xlabel("False Positive")
    plt.legend(loc=4)
    plt.savefig("ROC Curve")
    plt.show()
    '''

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
