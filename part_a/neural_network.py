import matplotlib.pyplot as plt

from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    train_data = load_train_csv(base_path)
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, train_data, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = self.g(inputs)
        out = F.sigmoid(out)
        out = self.h(out)
        out = F.sigmoid(out)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, train_data_dict, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    validation_accuracies = []
    train_accuracies = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + (lamb/2 * model.get_weight_norm())
            # loss = torch.sum((output - target) ** 2.)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

        validation_accuracies.append(valid_acc)

        train_acc = evaluate(model, zero_train_data, train_data_dict)
        train_accuracies.append(train_acc)

    return validation_accuracies, train_accuracies
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, train_data, valid_data, test_data = load_data()
    num_students, num_questions = zero_train_matrix.shape

    # Set model hyperparameters.
    k_values = [10, 50, 100, 200, 500]
    lamb_values = [0.001, 0.01, 0.1, 1]

    num_epoch = 40
    epoch_values = [i for i in range(0, num_epoch)]

    # Optimal: k = 100, lambda = 0.001, lr = 1e-2
    k_opt = 100
    lamb_opt = 0.001
    lr_opt = 1e-2

    model_opt = AutoEncoder(num_questions, k_opt)
    print("Valid Accuracies for k = ", k_opt, ", lambda = ", lamb_opt, ", Learning Rate:", lr_opt)
    validation_accuracies, train_accuracies = train(model_opt, lr_opt,
                                                    lamb_opt, train_matrix, zero_train_matrix,
                                                    train_data, valid_data, num_epoch)

    # Find test accuracy
    valid_accuracy = evaluate(model_opt, zero_train_matrix, valid_data)
    print("Valid Accuracy:", valid_accuracy)

    test_accuracy = evaluate(model_opt, zero_train_matrix, test_data)
    print("Test Accuracy:", test_accuracy)

    # Plot validation accuracies for best hyperparameters
    plt.plot(epoch_values, validation_accuracies)
    plt.plot(epoch_values, train_accuracies)
    plt.legend(["Validation Accuracies", "Train Accuracies"])
    plt.xlabel("Epoch Number")
    plt.ylabel("Accuracy")
    plt.title("Epoch vs Train/Validation Accuracy for k = 100 (no Regularization)")
    plt.savefig("Epoch vs Train and Validation Accuracy")
    plt.show()


if __name__ == "__main__":
    main()
