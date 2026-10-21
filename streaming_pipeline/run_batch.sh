#!/bin/bash
# Usage: bash run_batch.sh [days]   (default: 7)
DAYS=${1:-7}
python -m src.main --mode batch --days "$DAYS"
