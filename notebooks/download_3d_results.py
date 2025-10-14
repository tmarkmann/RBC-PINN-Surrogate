import json
import wandb
import pandas as pd
import os


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


# config
sweeps = [
    "sail-project/RBC-3D-FNO",
    "sail-project/RBC-3D-LRAN",
    "sail-project/RBC-3D-LSTM",
]
methods = ["3d-fno", "3d-lran", "3d-lstm"]

for sweep, method in zip(sweeps, methods):
    get_sweep_results(sweep, method, tag="final_result")
