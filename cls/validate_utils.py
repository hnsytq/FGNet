import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

def valid(device, model, test_loader):
    count = 0
    model.eval()
    y_pred_test = 0
    y_test = 0
    for idx, sample in enumerate(tqdm(test_loader)):
    # for inputs, labels, _, _ in test_loader:
        inputs, labels, _, _, _, _ = sample
        inputs = inputs.to(device)
        outputs = model(inputs, mode='val')
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test


def output_metric(tar, pre, mode=None):
    matrix = confusion_matrix(tar, pre)
    if mode is not None:
        return matrix
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA

def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA