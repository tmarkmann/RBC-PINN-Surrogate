import json
import wandb
import pandas as pd
import os
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)


def get_superres_results(project, method):
    # set your entity and project
    api = wandb.Api()
    srs = [f"sr{i}" for i in [1, 2, 4, 8, 16]]
    runs = api.runs(project, filters={"tags": {"$in": srs}})

    root = ROOT + "/results"
    os.makedirs(root, exist_ok=True)

    # Download
    for run in runs:
        sr = run.config["sr"]
        seed = run.config["seed"]
        for artifact in run.logged_artifacts():
            if artifact.type == "run_table" and "Table-Metrics" in artifact.name:
                # get artifact
                table_dir = artifact.download(root=f"{root}/artifacts/{run.id}")
                table_path = f"{table_dir}/test/Table-Metrics.table.json"
                # convert json to pd
                with open(table_path) as file:
                    json_dict = json.load(file)
                df = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])
                # write as csv
                outdir = f"{root}/{method}"
                os.makedirs(outdir, exist_ok=True)
                df.to_csv(f"{outdir}/sr{sr}-{seed}.csv", index=False)


def get_sweep_results(project, method, tag):
    # set your entity and project
    api = wandb.Api()
    runs = api.runs(project, filters={"tags": {"$in": [tag]}})

    root = "notebooks/results"
    os.makedirs(root, exist_ok=True)

    # Download
    for run in runs:
        ra = run.config.get("ra", 2500)
        seed = run.config.get("seed", 0)
        for artifact in run.logged_artifacts():
            if artifact.type == "run_table" and "Table-Metrics" in artifact.name:
                table_dir = artifact.download(root=f"{root}/artifacts/{run.id}")
                table_path = f"{table_dir}/test/Table-Metrics.table.json"
                table_type = "metrics"
            elif artifact.type == "run_table" and "Table-Nusselt" in artifact.name:
                table_dir = artifact.download(root=f"{root}/artifacts/{run.id}")
                table_path = f"{table_dir}/test/Table-Nusselt.table.json"
                table_type = "nusselt"
            elif artifact.type == "run_table" and "Table-Q-Profile" in artifact.name:
                table_dir = artifact.download(root=f"{root}/artifacts/{run.id}")
                table_path = f"{table_dir}/test/Table-Q-Profile.table.json"
                table_type = "q-profile"
            elif artifact.type == "run_table" and "Table-QP-Profile" in artifact.name:
                table_dir = artifact.download(root=f"{root}/artifacts/{run.id}")
                table_path = f"{table_dir}/test/Table-QP-Profile.table.json"
                table_type = "qp-profile"
            elif artifact.type == "run_table" and "Table-QP-Histogram" in artifact.name:
                table_dir = artifact.download(root=f"{root}/artifacts/{run.id}")
                table_path = f"{table_dir}/test/Table-QP-Histogram.table.json"
                table_type = "qp-histogram"
            else:
                continue

            # convert json to pd
            with open(table_path) as file:
                json_dict = json.load(file)
            df = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])
            # write as csv
            outdir = f"{root}/{method}/{table_type}"
            os.makedirs(outdir, exist_ok=True)
            df.to_csv(f"{outdir}/ra{ra}-{seed}.csv", index=False)


# Download Result runs
# sweeps = [
#     "sail-project/RBC-3D-FNO",
#     "sail-project/RBC-3D-LRAN",
#     "sail-project/RBC-3D-LSTM",
# ]
# methods = ["3d-fno", "3d-lran", "3d-lstm"]
#
# for sweep, method in zip(sweeps, methods):
#     get_sweep_results(sweep, method, tag="final_result")

# Download Superres runs
get_superres_results("sail-project/RBC-3D-FNO", "3d-fno-superres")
