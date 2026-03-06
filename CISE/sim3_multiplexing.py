import sys, os
sys.path.insert(0, '.')
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from src.noise_geometry import construct_sigma_A
from src.k_metric import compute_h_max, compute_k
from src.fisher_info import compute_fisher_information
from src.mi_estimator import estimate_mi_binning, project_onto_task_axis
from src.metrics_logger import MetricsLogger
SEED_BASE=42
DEFAULTS=dict(population_size=100,signal_amplitude=2.0,noise_variance=1.0,num_trials=3000,max_tasks=8)
def _ortho_axes(N,n_tasks,seed):
    rng=np.random.default_rng(seed)
    Q,_=np.linalg.qr(rng.standard_normal((N,n_tasks)))
    return [Q[:,i] for i in range(n_tasks)]
def _single(n_tasks,N,sig_str,noise_var,n_trials):
    axes=_ortho_axes(N,n_tasks,seed=SEED_BASE)
    sigma=construct_sigma_A(axes[0],noise_var,N)
    rng=np.random.default_rng(SEED_BASE+n_tasks)
    labels=rng.integers(0,2,size=n_trials)
    noise=rng.multivariate_normal(np.zeros(N),sigma,size=n_trials)
    responses=np.zeros((n_trials,N))
    for i,lab in enumerate(labels):
        signal=sum((sig_str/n_tasks)*ax*(lab*2-1) for ax in axes)
        responses[i]=signal+noise[i]
    proj=project_onto_task_axis(responses,axes[0])
    mi=estimate_mi_binning(proj,labels)
    fi=compute_fisher_information(sigma,axes[0],sig_str)
    k=compute_k(mi,compute_h_max(sigma,axes[0]))
    split=int(n_trials*0.8)
    sc=StandardScaler(); clf=LinearSVC(max_iter=3000,C=1.0)
    clf.fit(sc.fit_transform(responses[:split]),labels[:split])
    acc=clf.score(sc.transform(responses[split:]),labels[split:])
    return dict(n_tasks=n_tasks,MI=mi,FI=fi,K=k,accuracy=acc)
def run(params=None,save_dir="results"):
    p={**DEFAULTS,**(params or {})}
    logger=MetricsLogger("sim3_multiplexing",results_dir=save_dir)
    logger.save_parameters(p)
    results=[]
    for n in range(1,p["max_tasks"]+1):
        rec=_single(n,p["population_size"],p["signal_amplitude"],p["noise_variance"],p["num_trials"])
        results.append(rec); logger.log(rec)
        print(f"Tasks={n} | K={rec['K']:.3f} | MI={rec['MI']:.3f} | acc={rec['accuracy']:.3f}")
    metrics_path=logger.save_metrics()
    print(f"Metrics: {metrics_path}")
    return dict(results=results,metrics_path=metrics_path)
if __name__=="__main__":
    run()
