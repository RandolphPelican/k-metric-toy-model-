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
SEED_BASE=42
DEFAULTS=dict(population_size=80,signal_amplitude=2.0,noise_variance=1.0,num_trials=4000,n_axes_novice=1,n_axes_expert=5)
def _build(N,n_axes,sig_str,noise_var,n_trials,seed):
    rng=np.random.default_rng(seed)
    Q,_=np.linalg.qr(rng.standard_normal((N,max(n_axes,2))))
    axes=[Q[:,i] for i in range(n_axes)]
    sigma=construct_sigma_A(axes[0],noise_var,N)
    labels=rng.integers(0,2,size=n_trials)
    noise=rng.multivariate_normal(np.zeros(N),sigma,size=n_trials)
    responses=np.zeros((n_trials,N))
    for i,lab in enumerate(labels):
        signal=sum((sig_str/n_axes)*ax*(lab*2-1) for ax in axes)
        responses[i]=signal+noise[i]
    return responses,labels,sigma,axes
def _gen_acc(responses,labels,axes,sigma,N,sig_str,seed):
    rng=np.random.default_rng(seed+999)
    novel=rng.standard_normal(N)
    for ax in axes:
        novel-=np.dot(novel,ax)*ax
    novel=novel/np.linalg.norm(novel)
    n_test=1000
    tlab=rng.integers(0,2,size=n_test)
    tnoise=rng.multivariate_normal(np.zeros(N),sigma,size=n_test)
    tresp=np.array([sig_str*novel*(l*2-1)+tnoise[i] for i,l in enumerate(tlab)])
    sc=StandardScaler(); clf=LinearSVC(max_iter=3000,C=1.0)
    clf.fit(sc.fit_transform(responses),labels)
    return clf.score(sc.transform(tresp),tlab)
def run(params=None,save_dir="results"):
    p={**DEFAULTS,**(params or {})}
    logger=MetricsLogger("sim5_expert_novice",results_dir=save_dir)
    logger.save_parameters(p)
    systems={}
    for name,n_axes,seed in [("Novice",p["n_axes_novice"],SEED_BASE),("Expert",p["n_axes_expert"],SEED_BASE+1)]:
        resp,lab,sigma,axes=_build(p["population_size"],n_axes,p["signal_amplitude"],p["noise_variance"],p["num_trials"],seed)
        proj=project_onto_task_axis(resp,axes[0])
        mi=estimate_mi_binning(proj,lab)
        fi=compute_fisher_information(sigma,axes[0],p["signal_amplitude"])
        k=compute_k(mi,compute_h_max(sigma,axes[0]))
        split=int(p["num_trials"]*0.8)
        sc=StandardScaler(); clf=LinearSVC(max_iter=3000,C=1.0)
        clf.fit(sc.fit_transform(resp[:split]),lab[:split])
        tr_acc=clf.score(sc.transform(resp[split:]),lab[split:])
        gen=_gen_acc(resp,lab,axes,sigma,p["population_size"],p["signal_amplitude"],seed)
        rec=dict(system=name,n_axes=n_axes,MI=mi,FI=fi,K=k,training_accuracy=tr_acc,generalization_accuracy=gen)
        systems[name]=rec; logger.log(rec)
        print(f"{name}: K={k:.3f} | MI={mi:.3f} | train={tr_acc:.3f} | gen={gen:.3f}")
    metrics_path=logger.save_metrics()
    print(f"Metrics: {metrics_path}")
    return dict(systems=systems,metrics_path=metrics_path)
if __name__=="__main__":
    run()
