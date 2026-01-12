#!/bin/bash

BASEDIR=$(dirname "$0")
uv run wandb sweep --project RBC-3D-UNET "$BASEDIR/sweep_3d_unet.yaml"