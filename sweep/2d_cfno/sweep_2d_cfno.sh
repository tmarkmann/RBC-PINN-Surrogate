#!/bin/bash

BASEDIR=$(dirname "$0")
uv run wandb sweep --project RBC-2D-cFNO "$BASEDIR/control_mask.yaml"