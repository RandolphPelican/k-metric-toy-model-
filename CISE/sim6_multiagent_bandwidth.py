import sys, os
sys.path.insert(0, '.')
import numpy as np
from src.noise_geometry import get_task_axis, construct_sigma_A, construct_sigma_B
from src.population import get_task_means, generate_responses
from src.k_metric import compute_h_max, compute_k
from src.fisher_info import compute_fisher_information
from src.mi_estimator import estimate_mi_binning, project_onto_task_axis
from src.metrics_logger import MetricsLogger
SEED_BASE=42
DEFAULTS=dict(population_size=50,signal_amplitude=2.0,noise_variance=1.0,num_trials=2000,n_bandwidth_steps=15,bw_min=0.05,bw_max=1.0)
def _coord(responses,labels,task_axis,bw_frac,rng,n_agents=8):
    n_trials,N=responses.shape
    n_dims=max(1,int(bw_frac*N))
    keep=np.argsort(np.abs(task_axis))[::-1][:n_dims]
    agent_resps=[responses+rng.multivariate_normal(np.zeros(N),np.eye(N)*0.5,size=n_trials) for _ in range(n_agents)]
    correct=0
    for t in range(n_trials):
        votes=[]
        for ar in agent_resps:
            comp=np.zeros(N); comp[keep]=ar[t,keep]
            votes.append(1 if np.dot(comp,task_axis)>0 else 0)
        correct+=int((sum(votes)>n_agents/2)==labels[t])
    return correct/n_trials
def run(params=None,save_dir="results"):
    p={**DEFAULTS,**(params or {})}
    logger=MetricsLogger("sim6_multiagent_bandwidth",results_dir=save_dir)
    logger.save_parameters(p)
    N=p["population_size"]; sig_str=p["signal_amplitude"]; noise_var=p["noise_variance"]; n_trials=p["num_trials"]
    bws=np.linspace(p["bw_min"],p["bw_max"],p["n_bandwidth_steps"])
    task_axis=get_task_axis(N,seed=SEED_BASE)
    sigma_a=construct_sigma_A(task_axis,noise_var,N)
    sigma_b=construct_sigma_B(task_axis,noise_var,N)
    mu_0,mu_1=get_task_means(task_axis,sig_str,N)
    resp_a,lab_a=generate_responses(sigma_a,mu_0,mu_1,n_trials,seed=0)
    resp_b,lab_b=generate_responses(sigma_b,mu_0,mu_1,n_trials,seed=1)
    rng=np.random.default_rng(SEED_BASE)
    results=[]
    for bw in bws:
        for sname,resp,lab,sigma in [("A",resp_a,lab_a,sigma_a),("B",resp_b,lab_b,sigma_b)]:
            proj=project_onto_task_axis(resp,task_axis)
            mi=estimate_mi_binning(proj,lab)
            fi=compute_fisher_information(sigma,task_axis,sig_str)
            k=compute_k(mi,compute_h_max(sigma,task_axis))
            coord=_coord(resp,lab,task_axis,bw,rng)
            rec=dict(system=sname,bandwidth=float(bw),MI=mi,FI=fi,K=k,coordination_accuracy=coord)
            results.append(rec); logger.log(rec)
        print(f"BW={bw:.2f} | K_A={results[-2]['K']:.3f} K_B={results[-1]['K']:.3f} | coord_A={results[-2]['coordination_accuracy']:.3f} coord_B={results[-1]['coordination_accuracy']:.3f}")
    metrics_path=logger.save_metrics()
    print(f"Metrics: {metrics_path}")
    return dict(results=results,metrics_path=metrics_path)
if __name__=="__main__":
    run()
