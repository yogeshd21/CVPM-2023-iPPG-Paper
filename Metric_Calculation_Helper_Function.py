## Author: Yogesh Deshpande Aug 2021 - May 2023

from scipy.sparse import spdiags
import numpy as np
import scipy
import scipy.io
from scipy.signal import butter

def detrend(signal, Lambda):
    """detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended signal.
    """
    signal_length = signal.shape[0]

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal


def mag2db(mag):
    """Convert a magnitude to decibels (dB)
    If A is magnitude,
        db = 20 * log10(A)
    Parameters
    ----------
    mag : float or ndarray
        input magnitude or array of magnitudes
    Returns
    -------
    db : float or ndarray
        corresponding values in decibels
    """
    return 20. * np.log10(mag)

def calculate_HR(pxx_pred, frange_pred, fmask_pred, pxx_label, frange_label, fmask_label):
    pred_HR = np.take(frange_pred, np.argmax(np.take(pxx_pred, fmask_pred), 0))[0] * 60
    ground_truth_HR = np.take(frange_label, np.argmax(np.take(pxx_label, fmask_label), 0))[0] * 60
    return pred_HR, ground_truth_HR


def calculate_SNR(pxx_pred, f_pred, currHR, signal):
    currHR = currHR / 60
    f = f_pred
    pxx = pxx_pred
    gtmask1 = (f >= currHR - 0.1) & (f <= currHR + 0.1)
    gtmask2 = (f >= currHR * 2 - 0.1) & (f <= currHR * 2 + 0.1)
    sPower = np.sum(np.take(pxx, np.where(gtmask1 | gtmask2)))
    if signal == 'pulse':
        fmask2 = (f >= 0.75) & (f <= 4)
    else:
        fmask2 = (f >= 0.08) & (f <= 0.5)
    allPower = np.sum(np.take(pxx, np.where(fmask2 == True)))
    SNR_temp = mag2db(sPower / (allPower - sPower))
    return SNR_temp


def calculate_metric(predictions, labels, signal='pulse', window_size=360, fs=30, bpFlag=True):
    if signal == 'pulse':
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')  # 2.5 -> 1.7
    else:
        [b, a] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')

    data_len = len(predictions)
    HR_pred = []
    HR0_pred = []
    mySNR = []
    for j in range(0, data_len, window_size):
        if j == 0 and (j + window_size) > data_len:
            pred_window = predictions
            label_window = labels
        elif (j + window_size) > data_len:
            break
        else:
            pred_window = predictions[j:j + window_size]
            label_window = labels[j:j + window_size]
        if signal == 'pulse':
            pred_window = detrend(np.cumsum(pred_window), 100)
        else:
            pred_window = np.cumsum(pred_window)

        label_window = np.squeeze(label_window)
        if bpFlag:
            pred_window = scipy.signal.filtfilt(b, a, np.double(pred_window))

        pred_window = np.expand_dims(pred_window, 0)
        label_window = np.expand_dims(label_window, 0)
        # Predictions FFT
        f_prd, pxx_pred = scipy.signal.periodogram(pred_window, fs=fs, nfft=4 * window_size, detrend=False)
        if signal == 'pulse':
            fmask_pred = np.argwhere((f_prd >= 0.75) & (f_prd <= 2.5))  # regular Heart beat are 0.75*60 and 2.5*60
        else:
            fmask_pred = np.argwhere((f_prd >= 0.08) & (f_prd <= 0.5))  # regular Heart beat are 0.75*60 and 2.5*60
        pred_window = np.take(f_prd, fmask_pred)
        # Labels FFT
        f_label, pxx_label = scipy.signal.periodogram(label_window, fs=fs, nfft=4 * window_size, detrend=False)
        if signal == 'pulse':
            fmask_label = np.argwhere((f_label >= 0.75) & (f_label <= 2.5))  # regular Heart beat are 0.75*60 and 2.5*60
        else:
            fmask_label = np.argwhere((f_label >= 0.08) & (f_label <= 0.5))  # regular Heart beat are 0.75*60 and 2.5*60
        label_window = np.take(f_label, fmask_label)

        # MAE
        temp_HR, temp_HR_0 = calculate_HR(pxx_pred, pred_window, fmask_pred, pxx_label, label_window, fmask_label)
        temp_SNR = calculate_SNR(pxx_pred, f_prd, temp_HR_0, signal)
        HR_pred.append(temp_HR)
        HR0_pred.append(temp_HR_0)
        mySNR.append(temp_SNR)

    HR = np.array(HR_pred)
    HR0 = np.array(HR0_pred)
    mySNR = np.array(mySNR)

    MAE = np.mean(np.abs(HR - HR0))
    RMSE = np.sqrt(np.mean(np.square(HR - HR0)))
    meanSNR = np.nanmean(mySNR)
    return MAE, RMSE, meanSNR, HR0, HR

# Function to calculate Pearsons Correlation Coefficient
def pear(preds, labels):  # all variable operation
    sum_x = sum(preds)  # x
    sum_y = sum(labels)  # y
    sum_xy = sum(preds * labels)  # xy
    sum_x2 = sum(np.power(preds, 2))  # x^2
    sum_y2 = sum(np.power(labels, 2))  # y^2
    N = len(preds)
    pearson = (N * sum_xy - sum_x * sum_y) / (
        np.sqrt((N * sum_x2 - np.power(sum_x, 2)) * (N * sum_y2 - np.power(sum_y, 2))))

    return pearson


# Function for Authentication
def authentication(tp1, op_arr, gt_arr, no_candidates, window):
    z_arr = []
    temp = []
    sigtp1 = op_arr[(tp1 - 1) * window:tp1 * window]
    ans_person = 1
    max_corr = -2
    for i in range(no_candidates):
        chk = gt_arr[(i) * window:(i + 1) * window]
        corr = pear(sigtp1, chk)
        temp.append(corr)
        # print(max_corr, corr)
        if max_corr < corr:
            max_corr = corr
            ans_person = i + 1
    z_arr.append(temp)
    # print(z)
    return ans_person, z_arr

# Function for Shape Morphology Metrics
def signal_metrics(oparr, gtarr):

    ## Time domain Normalized Cross-Correlation
    norm_corr_time = sum(gtarr * oparr) / np.sqrt(sum(gtarr ** 2) * sum(oparr ** 2))

    ## Freq domain Normalized Cross-Correlation
    gt_freq = abs(np.fft.fft(gtarr))
    op_freq = abs(np.fft.fft(oparr))
    norm_corr_freq = sum(gt_freq * op_freq) / np.sqrt(sum(gt_freq ** 2) * sum(op_freq ** 2))

    ## Power spectral Normalized Cross-Correlation
    op_sf, op_power = scipy.signal.periodogram(oparr, 35)
    gt_sf, gt_power = scipy.signal.periodogram(gtarr, 35)
    norm_corr_power = sum(gt_power * op_power) / np.sqrt(sum(gt_power ** 2) * sum(op_power ** 2))

    return norm_corr_time, norm_corr_freq, norm_corr_power