import subprocess


param_groups = [
    "IQFT 10 --distance_metrics trace"
]


for i, params in enumerate(param_groups):
    print(f"\n===== Running iteration {i+1} parameter settings: {params} =====")
    cmd = f"python distance_metric_experiments.py {params}"
    subprocess.run(cmd, shell=True)

print("\n✅ All tasks completed!")