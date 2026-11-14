#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier
PYTHONPATH=/tests python3 -m verifier_pkg.verifier \
    --problem /app/problem.json \
    --design /app/design.json \
    --reward-out /logs/verifier/reward.txt \
    --metrics-out /logs/verifier/metrics.json
