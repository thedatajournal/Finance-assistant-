#!/bin/bash
# Keys come from the environment (docker run -e ...)
exec python -m src.main --mode realtime
