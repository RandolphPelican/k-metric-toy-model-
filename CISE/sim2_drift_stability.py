import sys, os
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.noise_geometry import get_task_axis, construct_sigma_A
from src.population import get_task_means, generate_responses
from src.permutation_test import compute_navigability
from src.mi_estimator import compute_raw_and_navigable_mi
from src.k_metric import compute_h_max, compute_k
from src.fisher_info import compute_fisher_information
from src.metrics_logger import MetricsLogger
SEED_TASK_AXIS=42; SEED_PERMUTATION=42
DEFAULTS=dict(population_size=100,signal_amplitude=2.0,noise_variance=1.0,num_trials=5000,drift_rate=0.15,n_epochs=30,n_permutations=500)
def _apply_drift(task_axis,sigma,drift_rate,rng,N):
    n_drift=max(1,int(drift_rate*N))
    drift_idx=rng.choice(N,size=n_drift,replace=False)
    perm=rng.permutation(drift_idx)
    new_axis=task_axis.copy()
    new_axis[drift_idx]=task_axis[perm]
    new_axis=new_axis/np.linalg.norm(new_axis)
    return new_axis,construct_sigma_A(new_axis,1.0,N)
def run(params=None,save_dir="results"):
    p={**DEFAULTS,**(params or {})}
    logger=MetricsLogger("sim2_drift_stability",results_dir=save_dir)
    logger.save_parameters(p)
    N=p["population_size"]; sig_str=p["signal_amplitude"]
    noise_var=p["noise_variance"]; n_trials=p["num_trials"]
    drift_rate=p["drift_rate"]; n_epochs=p["n_epochs"]; n_perm=p["n_permutations"]
    rng=np.random.default_rng(SEED_TASK_AXIS)
    task_axis=get_task_axis(N,seed=SEED_TASK_AXIS)
    sigma=construct_sigma_A(task_axis,noise_var,N)
    mu_0,mu_1=get_task_means(task_axis,sig_str,N)
    results=[]
    for epoch in range(n_epochs):
        if epoch>0:
            task_axis,sigma=_apply_drift(task_axis,sigma,drift_rate,rng,N)
            mu_0,mu_1=get_task_means(task_axis,sig_str,N)
        resp,lab=generate_responses(sigma,mu_0,mu_1,n_trials,seed=epoch)
        nav,thr,acc,frac,idx=compute_navigability(resp,lab,n_permutations=n_perm,seed=SEED_PERMUTATION)
        mi_raw,mi_nav=compute_raw_and_navigable_mi(resp,lab,nav,idx,task_axis)
        fi=compute_fisher_information(sigma,task_axis,sig_str)
        k=compute_k(mi_nav,compute_h_max(sigma,task_axis))
        rec=dict(epoch=epoch,MI_navigable=mi_nav,FI=fi,K=k,decoder_accuracy=acc)
        results.append(rec); logger.log(rec)
        print(f"Epoch {epoch:02d} | K={k:.3f} | MI={mi_nav:.3f} | acc={acc:.3f}")
    metrics_path=logger.save_metrics()
    print(f"Metrics: {metrics_path}")
    return dict(epoch_results=results,metrics_path=metrics_path)
if __name__=="__main__":
    run()
