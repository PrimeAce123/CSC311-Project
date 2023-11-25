# TODO: complete this file.
from utils import *
from item_response import sigmoid, update_theta_beta
from neural_network import AutoEncoder

from random import randrange
from sklearn.impute import KNNImputer
import numpy as np

import torch
from torch.autograd import Variable
import torch.optim as optim


def resample(data, indices=None):
    """Resample the dataset with replacement."""
    N = len(data['user_id'])
    bootstrap = {'user_id': [], 'question_id': [], 'is_correct': []}

    for _ in range(N):
        index = randrange(N)
        bootstrap['user_id'].append(data['user_id'][index])
        bootstrap['question_id'].append(data['question_id'][index])
        bootstrap['is_correct'].append(data['is_correct'][index])

    return bootstrap


def generate_matrix(data):
    """Generate a sparse matrix based on the given data."""
    matrix = np.full((542, 1774), np.nan)

    for i in range(len(data['user_id'])):
        user = data['user_id'][i]
        question = data['question_id'][i]
        correctness = data['is_correct'][i]
        matrix[user, question] = correctness

    return matrix


def knn_predict(matrix, data, k):
    """Return predictions made on data using knn impute by user."""
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix)
    predictions = mat[data['user_id'], data['question_id']]
    return predictions.tolist()


def irt_train(data, lr, iterations):
    """Train IRT model"""
    N = len(set(data["user_id"]))
    M = len(set(data["question_id"]))

    theta = np.zeros(N)
    beta = np.zeros(M)

    for i in range(iterations):
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta


def irt_predict(data, theta, beta):
    """Return predictions made on data using irt."""
    predictions = []

    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        predictions.append(sigmoid(x))

    return predictions


def nn_matrices(data):
    """Return the sparse matrix and zero sparse matrix used for the neural network."""
    matrix = generate_matrix(data)
    zero_matrix = matrix.copy()
    zero_matrix[np.isnan(matrix)] = 0
    zero_matrix = torch.FloatTensor(zero_matrix)
    matrix = torch.FloatTensor(matrix)
    return matrix, zero_matrix


def nn_train(model, lr, lamb, train_data, zero_train_data, num_epoch):
    """Train the neural network, where the objective also includes a regularizer."""
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + (lamb / 2 * model.get_weight_norm())
            # loss = torch.sum((output - target) ** 2.)
            loss.backward()

            optimizer.step()


def nn_predict(model, zero_matrix, data):
    """Returns predictions made on data using neural network model."""
    model.eval()

    predictions = []

    for i, u in enumerate(data["user_id"]):
        inputs = Variable(zero_matrix[u]).unsqueeze(0)
        output = model(inputs)

        pred = output[0][data["question_id"][i]].item()
        predictions.append(pred)

    return predictions


def evaluate(knn_pred, irt_pred, nn_pred, data):
    """Evaluate the predicted correctness based on the mean of the three models."""
    combined = np.array([knn_pred, irt_pred, nn_pred])
    means = np.mean(combined, axis=0)
    predictions = np.where(means >= 0.5, 1, 0)
    return np.mean(data['is_correct'] == predictions)


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    knn_matrix = generate_matrix(resample(train_data))
    irt_data = resample(train_data)
    nn_data = resample(train_data)
    nn_matrix, nn_zero_matrix = nn_matrices(nn_data)

    # k-nearest neighbors
    k = 11
    knn_val_pred = knn_predict(knn_matrix, val_data, k)
    knn_test_pred = knn_predict(knn_matrix, test_data, k)

    # item response theory
    lr = 0.02
    iterations = 100
    theta, beta = irt_train(irt_data, lr, iterations)
    irt_val_pred = irt_predict(val_data, theta, beta)
    irt_test_pred = irt_predict(test_data, theta, beta)

    # neural network
    k = 100
    lamb = 0.001
    lr = 1e-2
    num_epoch = 40
    num_questions = nn_zero_matrix.shape[1]
    model = AutoEncoder(num_questions, k)
    nn_train(model, lr, lamb, nn_matrix, nn_zero_matrix, num_epoch)
    nn_val_pred = nn_predict(model, nn_zero_matrix, val_data)
    nn_test_pred = nn_predict(model, nn_zero_matrix, test_data)

    # evaluate the ensemble's performance
    val_acc = evaluate(knn_val_pred, irt_val_pred, nn_val_pred, val_data)
    test_acc = evaluate(knn_test_pred, irt_test_pred, nn_test_pred, test_data)
    print("Validation Accuracy:", val_acc)
    print("Test Accuracy:", test_acc)


if __name__ == "__main__":
    main()
