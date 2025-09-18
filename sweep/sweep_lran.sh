#!/bin/bash

BASEDIR=$(dirname "$0")
uv run wandb sweep --project RBC-2D-LRAN "$BASEDIR/sweep_lran.yaml"