#!/bin/bash

BASEDIR=$(dirname "$0")
uv run wandb sweep --project RBC-2D-UNET "$BASEDIR/sweep_2d_unet.yaml"