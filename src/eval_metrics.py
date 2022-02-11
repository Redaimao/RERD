import numpy as np

from sklearn.metrics import accuracy_score, f1_score


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
    tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
    p = float(np.sum(true_label == 1))
    n = float(np.sum(true_label == 0))

    return (tp * (n / p) + tn) / (2 * n)


def eval_mosei_regression(y_pred, y_true, exclude_zero=False):
    test_preds = y_pred.view(-1).cpu().detach().numpy()
    test_truth = y_true.view(-1).cpu().detach().numpy()

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)
    test_preds_a3 = np.clip(test_preds, a_min=-1., a_max=1.)
    test_truth_a3 = np.clip(test_truth, a_min=-1., a_max=1.)

    mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    mult_a3 = multiclass_acc(test_preds_a3, test_truth_a3)

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
    non_zeros_binary_truth = (test_truth[non_zeros] > 0)
    non_zeros_binary_preds = (test_preds[non_zeros] > 0)

    non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
    non_zeros_f1_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average='weighted')

    binary_truth = (test_truth >= 0)
    binary_preds = (test_preds >= 0)

    acc2 = accuracy_score(binary_preds, binary_truth)
    f_score = f1_score(binary_truth, binary_preds, average='weighted')

    eval_results = {
        "Has0_acc_2": round(acc2, 4),
        "Has0_F1_score": round(f_score, 4),
        "Non0_acc_2": round(non_zeros_acc2, 4),
        "Non0_F1_score": round(non_zeros_f1_score, 4),
        "Mult_acc_5": round(mult_a5, 4),
        "Mult_acc_7": round(mult_a7, 4),
        "MAE": round(mae, 4),
        "Corr": round(corr, 4)
    }

    print("Has0_acc_2: ", round(acc2, 4))
    print("Has0_F1_score: ", round(f_score, 4))
    print("Non0_acc_2: ", round(non_zeros_acc2, 4))
    print("Non0_F1_score: ", round(non_zeros_f1_score, 4))
    print("Mult_acc_5:", round(mult_a5, 4))
    print("Mult_acc_7:", round(mult_a7, 4))
    print("MAE: ", round(mae, 4))
    print("Corr: ", round(corr, 4))

    print("-" * 50)
    return eval_results


def eval_mosi_classification(y_pred, y_true):
    """
    {
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2
    }
    """

    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()

    y_pred_3 = y_pred
    Mult_acc_3 = accuracy_score(y_pred_3, y_true)
    F1_score_3 = f1_score(y_true, y_pred_3, average='weighted')

    # two classes
    y_pred = np.array([[v[0], v[2]] for v in y_pred])

    # with 0 (<= 0 or > 0)
    y_pred_2 = np.argmax(y_pred, axis=1)
    y_true_2 = []
    for v in y_true:
        y_true_2.append(0 if v <= 1 else 1)
    y_true_2 = np.array(y_true_2)
    Has0_acc_2 = accuracy_score(y_pred_2, y_true_2)
    Has0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')

    # without 0 (< 0 or > 0)
    non_zeros = np.array([i for i, e in enumerate(y_true) if e != 1])
    y_pred_2 = y_pred[non_zeros]
    y_pred_2 = np.argmax(y_pred_2, axis=1)
    y_true_2 = y_true[non_zeros]
    Non0_acc_2 = accuracy_score(y_pred_2, y_true_2)
    Non0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')

    eval_results = {
        "Has0_acc_2": round(Has0_acc_2, 4),
        "Has0_F1_score": round(Has0_F1_score, 4),
        "Non0_acc_2": round(Non0_acc_2, 4),
        "Non0_F1_score": round(Non0_F1_score, 4),
        "Acc_3": round(Mult_acc_3, 4),
        "F1_score_3": round(F1_score_3, 4)
    }

    print("Has0_acc_2: ", round(Has0_acc_2, 4))
    print("Has0_F1_score: ", round(Has0_F1_score, 4))
    print("Non0_acc_2: ", round(Non0_acc_2, 4))
    print("Non0_F1_score: ", round(Non0_F1_score, 4))
    print("Acc_3: ", round(Mult_acc_3, 4))
    print("F1_score_3: ", round(F1_score_3, 4))

    print("-" * 50)

    return eval_results


def eval_mosei_senti(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)
    acc_score = accuracy_score(binary_truth, binary_preds)

    print("MAE: ", mae)
    print("Correlation Coefficient: ", corr)
    print("mult_acc_7: ", mult_a7)
    print("mult_acc_5: ", mult_a5)
    print("F1 score: ", f_score)
    print("Accuracy: ", accuracy_score(binary_truth, binary_preds))

    print("-" * 50)

    return [f_score, acc_score, mae, corr]


def eval_mosi(results, truths, exclude_zero=False):
    return eval_mosei_senti(results, truths, exclude_zero)

