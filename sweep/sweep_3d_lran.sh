#!/bin/bash

BASEDIR=$(dirname "$0")
uv run wandb sweep --project RBC-3D-LRAN "$BASEDIR/sweep_lran3d.yaml"