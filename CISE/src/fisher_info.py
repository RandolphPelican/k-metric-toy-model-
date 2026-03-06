"""
fisher_info.py
FI = signal_strength^2 * task_axis^T @ Sigma^{-1} @ task_axis
"""
import numpy as np

def compute_fisher_information(sigma, task_axis, signal_strength=1.0):
    task_axis_normalized = task_axis / np.linalg.norm(task_axis)
    try:
        sigma_inv = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        sigma_inv = np.linalg.pinv(sigma)
    fi = (signal_strength ** 2) * (task_axis_normalized @ sigma_inv @ task_axis_normalized)
    return float(fi)

def compute_fisher_information_sweep(sigma_fn, task_axis, param_values, signal_strength=1.0):
    return np.array([compute_fisher_information(sigma_fn(v), task_axis, signal_strength) for v in param_values])
