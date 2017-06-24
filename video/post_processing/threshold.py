"""
Author: Lluc Cardoner

Script for calculating the threshold of the predictions.
Formatting for MAP calculation with Mediaeval script.

"""
from __future__ import print_function

import h5py
import numpy as np
from matplotlib import pyplot as plt

model_num = 65
predictions = '/home/lluc/PycharmProjects/TFG/video/src/LSTM_{}_predictions.h5'.format(model_num)
pred_file = h5py.File(predictions, 'a')
output = '/home/lluc/PycharmProjects/TFG/trec_eval.8.1/LSTM_results/me16in_wien_video_LSTM{}.txt'.format(model_num)
out_file = open(output, 'a')


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def calculate_threshold(video_num, second_derivative_threshold=0.01, save=True, plot=False):
    """Calculates the threshold value for all the predicted labels for the segments of one video"""
    pred = []
    video_group = pred_file['back_label_mapping']['video_{}'.format(video_num)]
    for seg in video_group:
        pred.append(video_group[seg][()])
    pred = np.array(pred)  # array with all the predicted labels of the segments of one video
    pred = np.sort(pred)  # sort the array from lower to higher
    curve = savitzky_golay(pred, 11, 5)

    first_derivative = np.gradient(pred)
    second_derivative = np.gradient(first_derivative)

    x_threshold = 0
    for x, e in enumerate(second_derivative):
        if e > second_derivative_threshold:
            x_threshold = x
            break
    # print(x, pred[x])
    if plot:
        plt.figure(1)
        plt.subplot(131)
        plt.plot(pred)
        plt.subplot(132)
        plt.plot(curve)
        plt.subplot(133)
        plt.plot(range(second_derivative.size), second_derivative, c='r')
        t = [0.01] * len(second_derivative)
        plt.plot(range(len(t)), t)
        plt.show()

    if save:
        if '/thresholds' not in pred_file:
            thr = pred_file.create_group('thresholds')
        else:
            thr = pred_file['thresholds']
        thr.create_dataset('video_{}'.format(video_num), data=pred[x])

    return pred[x]


def create_submit_results(video_num, th=0.5):
    """Create the submit result file"""
    video_group = pred_file['back_label_mapping']['video_{}'.format(video_num)]
    for i in range(len(video_group.keys())):  # to make sure it is in order
        for seg in video_group:
            name = seg.split('_')
            if int(name[0]) == i:
                prob = video_group[seg][()]
                classification = 0 if prob < th else 1
                out_file.write('video_{},{},{},{}\n'.format(video_num, name[1], classification, prob))


for v in range(52, 52 + 26):
    threshold = calculate_threshold(v, second_derivative_threshold=0.01, save=False, plot=False)
    print(v, threshold)
    create_submit_results(v, th=threshold)
