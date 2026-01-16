#!/bin/bash

BASEDIR=$(dirname "$0")
uv run wandb sweep --project RBC-2D-AE "$BASEDIR/sweep_2d_ae.yaml"