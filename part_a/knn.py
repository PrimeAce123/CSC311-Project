import numpy
from sklearn.impute import KNNImputer
from sklearn import metrics
from utils import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import pairwise_distances


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
        self.user_factors = np.random.random((self.num_user, self.rank))
        self.item_factors = np.random.random((self.num_question, self.rank))

        self.train_record = []

        for _ in range(self.num_iter):
            self.user_factors = self.step(train, self.user_factors, self.item_factors)
            self.item_factors = self.step(train.transpose(), self.item_factors, self.user_factors)
            predictions = self.predict()
            train_error = self.compute_error(train, predictions)
            self.train_record.append(train_error)

        return self

    def step(self, train, unfixed_var, fixed_var):
        A = fixed_var.transpose().dot(fixed_var)
        A_inv = np.linalg.inv(A)
        b = train.dot(fixed_var)
        unfixed_var = b.dot(A_inv)

        return unfixed_var


    def predict(self):
        predictions = self.user_factors.dot(self.item_factors.transpose())
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


def plot_learning_curve(model):
    linewidth = 3
    plt.plot(model.train_record, label='Train', linewidth=linewidth)
    plt.xlabel("Iterations")
    plt.ylabel("Mean Squared Error")
    plt.legend(loc='best')
    plt.savefig("Matrix_Completion")


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    zero_train_matrix = sparse_matrix.copy()
    zero_train_matrix[np.isnan(sparse_matrix)] = 0

    matrix_complete = MatrixCompletion(num_iter=100, rank=300)
    matrix_complete.fit(zero_train_matrix)
    plot_learning_curve(matrix_complete)

    reconstruct_matrix = matrix_complete.user_factors.dot(matrix_complete.item_factors.transpose())
    print(reconstruct_matrix.shape)

    print(reconstruct_matrix)

    acc = sparse_matrix_evaluate(val_data, reconstruct_matrix)
    print("Accuracy:", acc)

    acc_test = sparse_matrix_evaluate(test_data, reconstruct_matrix)
    print("Accuracy", acc_test)

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
    '''


    '''
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
