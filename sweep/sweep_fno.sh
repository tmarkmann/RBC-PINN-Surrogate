#!/bin/bash

BASEDIR=$(dirname "$0")
uv run wandb sweep --project RayleighBenard-FNO "$BASEDIR/sweep_fno.yaml"