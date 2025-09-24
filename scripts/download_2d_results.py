import json
import wandb
import pandas as pd
import os


def get_sweep_results(sweep, method):
    # set your entity and project
    api = wandb.Api()
    sweep = api.sweep(sweep)
    runs = sweep.runs

    root = "notebooks/results"
    os.makedirs(root, exist_ok=True)

    # Download
    for run in runs:
        ra = run.config["ra"]
        seed = run.config["seed"]
        for metric in ["R-MSE", "RMSE"]:
            for artifact in run.logged_artifacts():
                if artifact.type == "run_table" and f"Table-{metric}" in artifact.name:
                    # get artifact
                    table_dir = artifact.download(root=f"{root}/artifacts/{run.id}")
                    table_path = f"{table_dir}/test/Table-{metric}.table.json"
                    # convert json to pd
                    with open(table_path) as file:
                        json_dict = json.load(file)
                    df = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])
                    # write as csv
                    outdir = f"{root}/{method}/{metric}"
                    os.makedirs(outdir, exist_ok=True)
                    df.to_csv(f"{outdir}/ra{ra}-{seed}.csv", index=False)


# config
sweeps = [
    "sail-project/RBC-2D-FNO/xlc8j5u1",
    "sail-project/RBC-2D-FNO/cpn8z6hq",
    "sail-project/RBC-2D-LRAN/c7fhz8pk",
]
methods = [
    "2d-fno2d",
    "2d-fno3d",
    "2d-lran",
]

for sweep, method in zip(sweeps, methods):
    get_sweep_results(sweep, method)
