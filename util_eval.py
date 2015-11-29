from __future__ import print_function
import numpy as np

def multiclass_log_loss(y_true, y_pred_matrix, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    idea from this post:
    http://www.kaggle.com/c/emc-data-science/forums/t/2149/is-anyone-noticing-difference-betwen-validation-and-leaderboard-error/12209#post12209

    Parameters
    ----------
    y_true : array, shape = [n_samples]
    y_pred_matrix : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred_matrix, eps, 1. - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred_matrix.shape)
    rows = actual.shape[0]
    actual[np.arange(rows), y_true.astype(int)] = 1
    vsota = np.sum(actual * np.log(predictions))
    return -1.0 / rows * vsota

def multiclass_accuracy(y_true, y_pred_matrix, eps=1e-15):
    """
    Multi class classificaiton accuracy
    """
    y_pred = np.array([np.argmax(preds) for preds in y_pred_matrix]).astype(int)
    return np.sum(y_pred == y_true.astype(int)) / float(len(y_true))

if __name__ == "__main__":
    print(multiclass_log_loss(np.array([0,1,2]),
                          np.array([[1,1,1],[0,1,0],[0,0,1]])))
    print(multiclass_accuracy(np.array([0,1,2]),
                              np.array([[1,1,1],[0,1,0],[0,0,1]])))
    print(multiclass_accuracy(np.array([0,1,2]),
                              np.array([[1,1,1],[0,1,0],[0,5,1]])))

    print(multiclass_log_loss(np.array([0,1,2]),
                          np.array([[1,0,0],[0,1,0],[0,0,1]])))
    print(multiclass_accuracy(np.array([0,1,2]),
                              np.array([[1,0,0],[0,1,0],[0,0,1]])))