import json
import wandb
import pandas as pd
import os
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)


def get_superres_results(sweep, method):
    # set your entity and project
    api = wandb.Api()
    sweep = api.sweep(sweep)
    runs = sweep.runs

    root = ROOT + "/results"
    os.makedirs(root, exist_ok=True)

    # Download
    for run in runs:
        sr = run.config["train_downsample"]
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


def get_sweep_results(sweep, method):
    # set your entity and project
    api = wandb.Api()
    sweep = api.sweep(sweep)
    runs = sweep.runs

    root = ROOT + "/results"
    os.makedirs(root, exist_ok=True)

    # Download
    for run in runs:
        ra = run.config["ra"]
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
                df.to_csv(f"{outdir}/ra{ra}-{seed}.csv", index=False)


def get_results(project, method, tag):
    # set your entity and project
    api = wandb.Api()
    runs = api.runs(project, filters={"tags": {"$in": [tag]}})

    root = ROOT + "/results"
    os.makedirs(root, exist_ok=True)

    # Artifact types to download
    names = [
        "Metrics",
        "Profiles",
    ]

    # Download
    for run in runs:
        ra = run.config["data"]["ra"]
        seed = run.config["seed"]
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
                    df.to_csv(f"{outdir}/ra{ra}-{seed}.csv", index=False)


# Download Result runs
projects = [
    "sail-project/RBC-2D-FNO",
    # "sail-project/RBC-2D-LRAN",
]
sweeps = [
    "sail-project/RBC-2D-FNO/fifkw3tu",
    "sail-project/RBC-2D-FNO/jg6h5fco",
    "sail-project/RBC-2D-LRAN/mzly1064",
]
methods = [
    # "2d-fno2d",
    "2d-fno3d",
    # "2d-lran",
]

# Download test runs
# for project, method in zip(projects, methods):
#     get_results(project, method, "revision_test")

# # Download Result sweep runs
# for sweep, method in zip(sweeps, methods):
#    get_sweep_results(sweep, method)

# Download Superres runs
get_superres_results("sail-project/RBC-2D-FNO/g4damf96", "2d-fno-superres")
