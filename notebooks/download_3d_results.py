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
        sr = run.config.get("sr", 1)
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

def get_sr_cons_results(project, method):
    # set your entity and project
    api = wandb.Api()
    runs = api.runs(project, filters={"tags": {"$in": ["sr-cons"]}})

    root = ROOT + "/results"
    os.makedirs(root, exist_ok=True)

    # Download
    for run in runs:
        seed = run.config["seed"]
        for artifact in run.logged_artifacts():
            if artifact.type == "run_table" and "Table" in artifact.name:
                # get artifact
                table_dir = artifact.download(root=f"{root}/artifacts/{run.id}")
                table_path = f"{table_dir}/test/Table.table.json"
                # convert json to pd
                with open(table_path) as file:
                    json_dict = json.load(file)
                df = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])
                # write as csv
                outdir = f"{root}/{method}"
                os.makedirs(outdir, exist_ok=True)
                df.to_csv(f"{outdir}/sr-cons-{seed}.csv", index=False)

def get_results(project, method, tag):
    # set your entity and project
    api = wandb.Api()
    runs = api.runs(project, filters={"tags": {"$in": [tag]}})

    root = ROOT + "/results"
    os.makedirs(root, exist_ok=True)

    # Artifact types to download
    names = [
        "Metrics",
        "Nusselt",
        "Q-Profile",
        "QP-Profile",
        "QP-Histogram",
        "Divergence",
        "Kinetic-Energy",
    ]

    # Download
    for run in runs:
        ra = run.config.get("ra", 2500)
        seed = run.config.get("seed", 0)
        run_root = f"{root}/artifacts/{run.id}"

        for artifact in run.logged_artifacts():
            for name in names:
                if artifact.type == "run_table" and f"Table-{name}" in artifact.name:
                    # get artifact and paths
                    table_dir = artifact.download(root=run_root)
                    table_path = f"{table_dir}/test/Table-{name}.table.json"
                    table_type = name.lower()

                    # convert json to pd
                    with open(table_path) as file:
                        json_dict = json.load(file)
                    df = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])

                    # write as csv
                    outdir = f"{root}/{method}/{table_type}"
                    os.makedirs(outdir, exist_ok=True)

                    outfile = f"{outdir}/ra{ra}-{seed}.csv"
                    if os.path.exists(outfile):
                        print(f"Warning: Overwriting existing file {outfile}")
                    df.to_csv(outfile, index=False)


# Download Result runs
sweeps = [
    "sail-project/RBC-3D-FNO",
    "sail-project/RBC-3D-LRAN",
    # "sail-project/RBC-3D-LSTM",
]
methods = [
    "3d-fno",
    "3d-lran",
    # "3d-lstm",
]
# for sweep, method in zip(sweeps, methods):
#     get_results(sweep, method, tag="revision_test")

# Download Superres runs
get_superres_results("sail-project/RBC-3D-FNO", "3d-fno-superres")
get_sr_cons_results("sail-project/RBC-3D-FNO", "3d-fno-superres")
