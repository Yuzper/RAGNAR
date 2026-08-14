#!/usr/bin/env bash
# fetchjob.sh — fetch files matching *<filename>* from the HPC cluster.
# Usage: bash fetchjob.sh <filename>
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/venv/Scripts/python" "$SCRIPT_DIR/fetchjob.py" "$@"
