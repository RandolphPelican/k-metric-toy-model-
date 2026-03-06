import sys, os
sys.path.insert(0, ".")  #(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.noise_geometry import get_task_axis, construct_sigma_A, construct_sigma_B, verify_matching
from src.population import get_task_means, generate_responses
from src.permutation_test import compute_navigability
from src.mi_estimator import compute_raw_and_navigable_mi
from src.k_metric import compute_h_max, compute_k
from src.fisher_info import compute_fisher_information
from src.behavioral_sim import simulate_two_step_task
from src.metrics_logger import MetricsLogger

SEED_TASK_AXIS=42; SEED_NOISE=0; SEED_TRIALS=1; SEED_PERMUTATION=42
DEFAULTS=dict(population_size=50,signal_amplitude=2.0,noise_variance=1.0,num_trials=10000,n_permutations=1000)

def run(params=None,save_dir="results"):
    p={**DEFAULTS,**(params or {})}
    logger=MetricsLogger("sim1_noise_alignment",results_dir=save_dir)
    logger.save_parameters({**p,"seed_task_axis":SEED_TASK_AXIS,"seed_noise":SEED_NOISE,"seed_trials":SEED_TRIALS,"seed_permutation":SEED_PERMUTATION})
    N=p["population_size"]; sig_str=p["signal_amplitude"]; noise_var=p["noise_variance"]; n_trials=p["num_trials"]; n_perm=p["n_permutations"]
    task_axis=get_task_axis(N,seed=SEED_TASK_AXIS)
    sigma_a=construct_sigma_A(task_axis,noise_var,N)
    sigma_b=construct_sigma_B(task_axis,noise_var,N)
    verify_matching(sigma_a,sigma_b)
    mu_0,mu_1=get_task_means(task_axis,sig_str,N)
    resp_a,lab_a=generate_responses(sigma_a,mu_0,mu_1,n_trials,seed=SEED_NOISE)
    resp_b,lab_b=generate_responses(sigma_b,mu_0,mu_1,n_trials,seed=SEED_TRIALS)
    nav_a,thr_a,acc_a,frac_a,idx_a=compute_navigability(resp_a,lab_a,n_permutations=n_perm,seed=SEED_PERMUTATION)
    nav_b,thr_b,acc_b,frac_b,idx_b=compute_navigability(resp_b,lab_b,n_permutations=n_perm,seed=SEED_PERMUTATION)
    mi_raw_a,mi_nav_a=compute_raw_and_navigable_mi(resp_a,lab_a,nav_a,idx_a,task_axis)
    mi_raw_b,mi_nav_b=compute_raw_and_navigable_mi(resp_b,lab_b,nav_b,idx_b,task_axis)
    fi_a=compute_fisher_information(sigma_a,task_axis,sig_str)
    fi_b=compute_fisher_information(sigma_b,task_axis,sig_str)
    h_max_a=compute_h_max(sigma_a,task_axis); h_max_b=compute_h_max(sigma_b,task_axis)
    k_a=compute_k(mi_nav_a,h_max_a); k_b=compute_k(mi_nav_b,h_max_b)
    beh_a=simulate_two_step_task(resp_a,lab_a,sigma_a,task_axis,seed=42)
    beh_b=simulate_two_step_task(resp_b,lab_b,sigma_b,task_axis,seed=42)
    for sys,mi_r,mi_n,fi,k,beh,frac in [("A",mi_raw_a,mi_nav_a,fi_a,k_a,beh_a,frac_a),("B",mi_raw_b,mi_nav_b,fi_b,k_b,beh_b,frac_b)]:
        logger.log(dict(system=sys,MI_raw=mi_r,MI_navigable=mi_n,FI=fi,K=k,behavioral_accuracy=beh,navigable_fraction=frac))
    metrics_path=logger.save_metrics()
    print(f"K_A={k_a:.4f} K_B={k_b:.4f} beh_A={beh_a:.4f} beh_B={beh_b:.4f}")
    print(f"Metrics: {metrics_path}")
    return dict(MI_raw_A=mi_raw_a,MI_raw_B=mi_raw_b,MI_nav_A=mi_nav_a,MI_nav_B=mi_nav_b,FI_A=fi_a,FI_B=fi_b,K_A=k_a,K_B=k_b,beh_A=beh_a,beh_B=beh_b,metrics_path=metrics_path)

if __name__=="__main__":
    run()
