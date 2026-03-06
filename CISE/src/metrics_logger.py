"""
metrics_logger.py
Logs metrics to structured JSON output.
"""
import json
import os
import numpy as np
from datetime import datetime

class MetricsLogger:
    def __init__(self, sim_name, results_dir="results"):
        self.sim_name = sim_name
        self.sim_dir = os.path.join(results_dir, sim_name)
        self.figures_dir = os.path.join(self.sim_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)
        self.records = []

    def log(self, record: dict):
        self.records.append(record)

    def save_metrics(self, extra: dict = None):
        payload = {
            "sim_name": self.sim_name,
            "timestamp": datetime.utcnow().isoformat(),
            "n_records": len(self.records),
            "records": self.records,
        }
        if extra:
            payload.update(extra)
        path = os.path.join(self.sim_dir, "metrics.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=_json_safe)
        return path

    def save_parameters(self, params: dict):
        path = os.path.join(self.sim_dir, "parameters.json")
        with open(path, "w") as f:
            json.dump(params, f, indent=2, default=_json_safe)
        return path

    def figures_path(self, filename):
        return os.path.join(self.figures_dir, filename)

    def summary_table(self):
        return self.records

def _json_safe(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} not JSON serializable")
