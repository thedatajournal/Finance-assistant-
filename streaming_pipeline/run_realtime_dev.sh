#!/bin/bash
# Loads keys from a local .env file, then streams live news
set -a
[ -f .env ] && source .env
set +a
python -m src.main --mode realtime
