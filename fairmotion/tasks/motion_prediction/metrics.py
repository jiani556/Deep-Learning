# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import cv2
from fairmotion.ops import conversions


def euler_diff(predictions, targets):
    """
    Computes the Euler angle error as in previous work, following
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/translate.py#L207
    Args:
        predictions: np array of predicted joint angles represented as rotation matrices, i.e. in shape
          (..., n_joints, 3, 3)
        targets: np array of same shape as `predictions`

    Returns:
        The Euler angle error an np array of shape (..., )
    """
    assert predictions.shape[-1] == 3 and predictions.shape[-2] == 3
    assert targets.shape[-1] == 3 and targets.shape[-2] == 3
    n_joints = predictions.shape[-3]

    ori_shape = predictions.shape[:-3]
    preds = np.reshape(predictions, [-1, 3, 3])
    targs = np.reshape(targets, [-1, 3, 3])

    euler_preds = conversions.R2E(preds)  # (N, 3)
    euler_targs = conversions.R2E(targs)  # (N, 3)

    # reshape to (-1, n_joints*3) to be consistent with previous work
    euler_preds = np.reshape(euler_preds, [-1, n_joints * 3])
    euler_targs = np.reshape(euler_targs, [-1, n_joints * 3])

    # l2 error on euler angles
    idx_to_use = np.where(np.std(euler_targs, 0) > 1e-4)[0]
    euc_error = np.power(
        euler_targs[:, idx_to_use] - euler_preds[:, idx_to_use], 2,
    )
    euc_error = np.sqrt(np.sum(euc_error, axis=1))  # (-1, ...)

    # reshape to original
    return np.reshape(euc_error, ori_shape)

#TODO metrics
def joint_angle_diff(predictions, targets):
    """
       Computes the angular distance between the target and predicted rotations.
       following https://github.com/eth-ait/spl/blob/master/metrics/motion_metrics.py
       This essentially computes || log(R_diff) || where R_diff is the
       difference rotation between prediction and target.
       Input: predictionsnp array of predicted joint angles represented as rotation matrices, i.e. in shape
          (..., n_joints, 3, 3)
            targets: np array of same shape as `predictions`
       Returns:
           The geodesic distance for each joint as an np array of shape (..., n_joints)
       """
    assert predictions.shape[-1] == predictions.shape[-2] == 3
    assert targets.shape[-1] == targets.shape[-2] == 3

    ori_shape = predictions.shape[:-2]
    preds = np.reshape(predictions, [-1, 3, 3])
    targs = np.reshape(targets, [-1, 3, 3])

    # compute R1 * R2.T, if prediction and target match, this will be the identity matrix
    r = np.matmul(preds, np.transpose(targs, [0, 2, 1]))

    # convert `r` to angle-axis representation and extract the angle
    # which is our measure of difference between
    # the predicted and target orientations
    angles = []
    for i in range(r.shape[0]):
        aa, _ = cv2.Rodrigues(r[i])
        angles.append(np.linalg.norm(aa))
    angles = np.array(angles)

    return np.reshape(angles, ori_shape)

def positional_diff(predictions, targets):
    assert predictions.shape[-1] == 3 and targets.shape[-1] == 3
    pos_diff = np.mean(np.sqrt(np.sum((predictions - targets)**2, axis = -1)), axis = -1)    
    return pos_diff

def pck(predictions, targets, threshold):
    """
    PCK indicates the percentage of pred joints lying within
    a spherical threshold around tgt joints positions
    
    predictions: np array of predicted joint positions, i.e. in shape
                (..., n_joints, 3)
    targets: np array of same shape as `predictions`
    threshold: diff threshold

    """
    assert predictions.shape[-1] == 3 and targets.shape[-1] == 3
    pos_diff = np.sqrt(np.sum((predictions - targets) ** 2, axis=-1))
    pck = np.mean(np.array(pos_diff <= threshold, dtype=pos_diff.dtype), axis=-1)
    return pck

def pck_diff(predictions, targets):
    """
    Calculate PCK AUC given pred joint pos and tgt joint pos

    """
    thresholds = [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    pcks = [pck(predictions, targets, threshold) for threshold in thresholds]
    pck_diffs = [np.mean(pck) for pck in pcks]
    return calculate_pck_auc(pck_diffs, thresholds)

def calculate_pck_auc(pck_diffs, thresholds):
    """
    pck_diffs: PCK values
    thresholds: corresponding thresholds when calculating PCK values

    """
    auc = 0
    for i in range(len(pck_diffs) - 1):
        auc += (pck_diffs[i] + pck_diffs[i + 1]) / 2 * (thresholds[i + 1] - thresholds[i])
    return auc / (thresholds[-1] - thresholds[0])

#below are optional
def ps_entropy(predictions, targets):
    """
    Power spectrum entropy
    """
    pass

def ps_kl_divergence(predictions, targets):
    """
    Power spectrum KL divergence
    """
    pass