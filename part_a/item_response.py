from utils import *

import numpy as np

import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0
    for i in range(len(data["is_correct"])):
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        is_correct = data["is_correct"][i]
        
        theta_u = theta[user_id]
        beta_i = beta[question_id]
        
        p = sigmoid(theta_u - beta_i)

        if is_correct:
            log_lklihood += np.log(p)
        else:
            log_lklihood += np.log(1 - p)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################
    theta_grad = np.zeros_like(theta)
    beta_grad = np.zeros_like(beta)

    for i in range(len(data["is_correct"])):
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        is_correct = data["is_correct"][i]
        
        theta_u = theta[user_id]
        beta_i = beta[question_id]
        
        p = sigmoid(theta_u - beta_i)
        
        grad = is_correct - p
        
        theta_grad[user_id] += grad
        beta_grad[question_id] -= grad
    
    theta += lr * theta_grad
    beta += lr * beta_grad
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    N = len(set(data["user_id"]))
    M = len(set(data["question_id"]))

    theta = np.zeros(N)
    beta = np.zeros(M)

    val_acc_lst = []

    train_lld_list = []
    val_lld_list = []

    for i in range(iterations):
        neg_lld_train = neg_log_likelihood(data, theta=theta, beta=beta)
        neg_lld_val = neg_log_likelihood(val_data, theta=theta, beta=beta)
        #score = evaluate(data=val_data, theta=theta, beta=beta)
        #val_acc_lst.append(score)
        # print("NLLK: {} \t Score: {}".format(neg_lld, score))

        train_lld_list.append(-neg_lld_train)
        val_lld_list.append(-neg_lld_val)

        theta, beta = update_theta_beta(data, lr, theta, beta)

    # plot validation and training log-likelihood
    plt.figure(figsize=(10, 6))
    plt.plot(train_lld_list, label='Training log-likelihood')
    plt.plot(val_lld_list, label='Validation log-likelihood')
    plt.xlabel('Iterations')
    plt.ylabel('Log-likelihood')
    plt.title('Training and Validation Log-likelihoods over Iterations')
    plt.legend()

    plt.savefig("lld_vs_iteration.png")

    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def plot_question_difficulty(theta, beta, question_ids):
    theta_range = np.linspace(min(theta), max(theta), 100)
    
    plt.figure(figsize=(10, 6))
    
    for question_id in question_ids:
        beta_q = beta[question_id]
        p_correct = sigmoid(theta_range - beta_q)
        plt.plot(theta_range, p_correct, label=f'Question {question_id + 1}')

    plt.title('Probability of Correct Response for Different Questions')
    plt.xlabel('Student Ability (Theta)')
    plt.ylabel('Probability of Correct Response')
    plt.legend()
    plt.savefig("question_difficulty.png")


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################

    # tunning
    lr = 0.02
    iterations = 100

    theta, beta, val_acc_lst = irt(train_data, val_data, lr, iterations)

    val_accuracy = evaluate(val_data, theta, beta)
    test_accuracy = evaluate(test_data, theta, beta)

    print("Validation accuracy: {}".format(val_accuracy))
    print("Test accuracy: {}".format(test_accuracy))

    # Validation accuracy: 0.7056167090036692
    # Test accuracy: 0.7090036692068868

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # Implement part (d)                                                #

    question_ids_to_plot = [0, 1, 2]
    plot_question_difficulty(theta, beta, question_ids_to_plot)

    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
