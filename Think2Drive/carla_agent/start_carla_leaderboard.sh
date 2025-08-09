#!/bin/bash
set -euo pipefail # Exit immediately if a command exits with a non-zero status.
                 # Treat unset variables as an error when substituting.
                 # Pipelines return the exit status of the last command that failed.

# === Configuration ===
# This script runs the CARLA leaderboard evaluator.
# Assumes execution from: [...]/master_thesis_rl/carla_agent/

# === Argument Handling ===
if [[ "$#" -ne 7 ]]; then
    echo "Usage: $0 <route_file> <checkpoint_path> <client_port> <tm_port> <random_seed> <ipc_file> <is_evaluation>"
    echo "  <is_evaluation>: 'True' for official evaluation setup, 'False' for custom/training setup."
    echo "Example: $0 path/to/routes.xml path/to/model.pth 2000 8000 42 /tmp/ipc_file True"
    exit 1
fi

# Assign command-line arguments to variables
ARG_ROUTE_FILE="$1"         # Path to the routes definition file (e.g., routes_devtest.xml)
ARG_CHECKPOINT_PATH="$2"    # Path to the agent's model checkpoint file
ARG_CLIENT_PORT="$3"        # CARLA client RPC port
ARG_TM_PORT="$4"            # CARLA Traffic manager port
ARG_RANDOM_SEED="$5"        # Seed for Traffic Manager randomness
ARG_IPC_FILE="$6"           # Path for Inter-Process Communication file (used by agent/evaluator)
ARG_IS_EVALUATION="$7"      # Flag ('True' or 'False') to determine leaderboard setup

# === Environment Setup ===
echo "Initializing Conda..."
# Source Conda's shell functions
eval "$(conda shell.bash hook)"
# Activate the required Python environment
# conda activate carl
conda activate carl
echo "Activated Conda environment: carl"

# === Path Definitions ===
# Define core CARLA installation path (relative to this script's assumed location)
export CARLA_ROOT="PATH_TO_CARLA"

# Select Leaderboard and ScenarioRunner paths based on evaluation mode
if [[ "${ARG_IS_EVALUATION}" == "True" ]]; then
    echo "Using ORIGINAL leaderboard setup for evaluation."
    LEADERBOARD_BASE_DIR="../../CARLA/original_leaderboard"
    LEADERBOARD_REPETITIONS=1
    EXTRA_PYTHON_ARGS="" # No extra args needed for official evaluation
else
    echo "Using CUSTOM leaderboard setup."
    LEADERBOARD_BASE_DIR="../../CARLA/custom_leaderboard"
    LEADERBOARD_REPETITIONS=5 # Higher repetitions for training/testing
    # Enable rendering in custom/non-evaluation mode
    EXTRA_PYTHON_ARGS="--no_rendering_mode False --gym_port 0"
fi

export SCENARIO_RUNNER_ROOT="${LEADERBOARD_BASE_DIR}/scenario_runner"
export LEADERBOARD_ROOT="${LEADERBOARD_BASE_DIR}/leaderboard"

# Construct and export PYTHONPATH
# Order matters: Add project-specific paths first, then CARLA API
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":"${CARLA_ROOT}/PythonAPI"
echo "PYTHONPATH set."
# echo "DEBUG: PYTHONPATH=${PYTHONPATH}" # Uncomment to debug

# === Leaderboard Configuration ===
# Environment variables potentially used by the leaderboard or agent scripts
export CHECKPOINT_PATH="${ARG_CHECKPOINT_PATH}" # Export checkpoint path for agent access
export IPC_FILE="${ARG_IPC_FILE}"              # Export IPC file path
export NO_CARS=0                               # Example env var for leaderboard (0 often means use scenario defaults)
export EVALUATION="${ARG_IS_EVALUATION}"       # Export evaluation flag for agent access

# === Execute Leaderboard ===
LEADERBOARD_SCRIPT="${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py"
AGENT_SCRIPT="carla_agent.py" # The agent implementation to test

echo "Starting CARLA Leaderboard Evaluator..."
echo "  Route File: ${ARG_ROUTE_FILE}"
echo "  Checkpoint: ${ARG_CHECKPOINT_PATH}"
echo "  Client Port: ${ARG_CLIENT_PORT}"
echo "  TM Port: ${ARG_TM_PORT}"
echo "  TM Seed: ${ARG_RANDOM_SEED}"
echo "  Repetitions: ${LEADERBOARD_REPETITIONS}"
echo "  Evaluation Mode: ${ARG_IS_EVALUATION}"
[[ -n "${EXTRA_PYTHON_ARGS}" ]] && echo "  Extra Args: ${EXTRA_PYTHON_ARGS}"

# Execute the leaderboard evaluator script using Python
# -u: force unbuffered stdout/stderr
python -u "${LEADERBOARD_SCRIPT}" \
    --routes "${ARG_ROUTE_FILE}" \
    --agent "${AGENT_SCRIPT}" \
    --checkpoint "${ARG_CHECKPOINT_PATH}" \
    --track MAP \
    --port "${ARG_CLIENT_PORT}" \
    --traffic-manager-port "${ARG_TM_PORT}" \
    --traffic-manager-seed "${ARG_RANDOM_SEED}" \
    --timeout 1600.0 \
    --repetitions "${LEADERBOARD_REPETITIONS}" \
    ${EXTRA_PYTHON_ARGS} # Add extra arguments if defined (e.g., for non-evaluation)

echo "Leaderboard evaluation finished."