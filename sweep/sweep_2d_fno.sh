#!/bin/bash

BASEDIR=$(dirname "$0")
uv run wandb sweep --project RBC-2D-FNO "$BASEDIR/sweep_fno_ra_3D.yaml"