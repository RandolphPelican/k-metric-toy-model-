import sys, os
sys.path.insert(0, '.')
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from src.noise_geometry import get_task_axis, construct_sigma_A
from src.k_metric import compute_h_max, compute_k
from src.fisher_info import compute_fisher_information
from src.mi_estimator import estimate_mi_binning, project_onto_task_axis
from src.metrics_logger import MetricsLogger
SEED_TASK_AXIS=42
DEFAULTS=dict(population_size=50,noise_variance=1.0,num_trials=3000,gain_min=0.1,gain_max=3.0,n_gain_steps=20)
def run(params=None,save_dir="results"):
    p={**DEFAULTS,**(params or {})}
    logger=MetricsLogger("sim4_attention_gain",results_dir=save_dir)
    logger.save_parameters(p)
    N=p["population_size"]; noise_var=p["noise_variance"]; n_trials=p["num_trials"]
    gains=np.linspace(p["gain_min"],p["gain_max"],p["n_gain_steps"])
    task_axis=get_task_axis(N,seed=SEED_TASK_AXIS)
    rng=np.random.default_rng(SEED_TASK_AXIS)
    results=[]
    for g in gains:
        sig_str=g*2.0
        sigma_eff=(g**2)*construct_sigma_A(task_axis,noise_var,N)
        mu_1=sig_str*task_axis
        labels=rng.integers(0,2,size=n_trials)
        noise=rng.multivariate_normal(np.zeros(N),sigma_eff,size=n_trials)
        responses=np.zeros((n_trials,N))
        for i,lab in enumerate(labels):
            responses[i]=(mu_1 if lab==1 else np.zeros(N))+noise[i]
        proj=project_onto_task_axis(responses,task_axis)
        mi=estimate_mi_binning(proj,labels)
        fi=compute_fisher_information(sigma_eff,task_axis,sig_str)
        h_max=compute_h_max(sigma_eff,task_axis)
        k=compute_k(mi,h_max) if h_max>0 else 0.0
        split=int(n_trials*0.8)
        sc=StandardScaler(); clf=LinearSVC(max_iter=3000,C=1.0)
        clf.fit(sc.fit_transform(responses[:split]),labels[:split])
        acc=clf.score(sc.transform(responses[split:]),labels[split:])
        rec=dict(gain=float(g),MI=mi,FI=fi,K=k,accuracy=acc)
        results.append(rec); logger.log(rec)
        print(f"gain={g:.2f} | K={k:.3f} | MI={mi:.3f} | acc={acc:.3f}")
    metrics_path=logger.save_metrics()
    print(f"Metrics: {metrics_path}")
    return dict(results=results,metrics_path=metrics_path)
if __name__=="__main__":
    run()
