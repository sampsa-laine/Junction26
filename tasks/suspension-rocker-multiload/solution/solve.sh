#!/bin/bash
set -euo pipefail

PYTHONPATH=/solution:/opt python3 -m oracle_pkg.oracle_solver \
    --problem /app/problem.json \
    --design /app/design.json

cad-sim status --ascii
