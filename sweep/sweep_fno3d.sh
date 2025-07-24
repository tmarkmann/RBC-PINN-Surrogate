#!/bin/bash

BASEDIR=$(dirname "$0")
uv run wandb sweep --project RayleighBenard-3D-FNO "$BASEDIR/sweep_fno3d.yaml"