# from starter_code.part_a.utils import *

import part_a.utils


def main():
    """ Returns the majority vote.

    This code serves as an example how to load the data and
    submit your result to kaggle.

    :return: None
    """
    # Load the train data.
    train_data = part_a.utils.load_train_csv("data")

    correct_question_map = {}
    total_question_map = {}

    # Count how many questions were correct.
    for i, q in enumerate(train_data["question_id"]):
        if q in correct_question_map:
            if train_data["is_correct"][i] == 1:
                correct_question_map[q] += 1
            total_question_map[q] += 1
        else:
            if train_data["is_correct"][i] == 1:
                correct_question_map[q] = 1
            total_question_map[q] = 1

    # Load the validation data.
    valid_data = part_a.utils.load_valid_csv("data")
    predictions = []
    for i, q in enumerate(valid_data["question_id"]):
        ratio = correct_question_map[q] / float(total_question_map[q])
        # If the question was answered correctly more than half
        # of the times, predict correct.
        if ratio >= 0.5:
            predictions.append(1.)
        else:
            predictions.append(0.)
    # Evaluate your model using the new prediction.
    acc = part_a.utils.evaluate(valid_data, predictions)
    print("Validation Accuracy: {}".format(acc))

    # Load the public test data.
    test_data = part_a.utils.load_public_test_csv("data")
    predictions = []
    for i, q in enumerate(test_data["question_id"]):
        ratio = correct_question_map[q] / float(total_question_map[q])
        if ratio >= 0.5:
            predictions.append(1.)
        else:
            predictions.append(0.)
    acc = part_a.utils.evaluate(test_data, predictions)
    print("Test Accuracy: {}".format(acc))

    # Load the private test dataset.
    private_test = part_a.utils.load_private_test_csv("data")
    predictions = []
    for i, q in enumerate(private_test["question_id"]):
        ratio = correct_question_map[q] / float(total_question_map[q])
        if ratio >= 0.5:
            predictions.append(1.)
        else:
            predictions.append(0.)
    private_test["is_correct"] = predictions
    # Save your predictions to make it ready to submit to Kaggle.
    part_a.utils.save_private_test_csv(private_test)
    return


if __name__ == "__main__":
    main()
