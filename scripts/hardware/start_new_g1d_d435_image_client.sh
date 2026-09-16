#!/usr/bin/env bash
set -euo pipefail

# Dedicated launcher for the new G1D with a D435 RGB head camera.
# Keep image_client_g1.py defaults unchanged so other robot launch paths are unaffected.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_IP="${XR_TELEOP_ROBOT_IP:-10.48.61.116}"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/image_client_g1.py" --host "$ROBOT_IP" "$@"
