#!/bin/bash

BASEDIR=$(dirname "$0")
uv run wandb sweep --project RBC-3D-FNO "$BASEDIR/sweep_3d_fno_sr.yaml"