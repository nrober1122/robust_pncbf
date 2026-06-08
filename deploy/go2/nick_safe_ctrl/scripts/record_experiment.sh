#!/usr/bin/env bash
#
# record_experiment.sh -- record a safe_ctrl_node run with its config snapshot,
# then auto-analyze it for collisions when you stop.
#
# Workflow (flexible, no strict regimen):
#   1. Launch the node however you like (real or dry-run), in another terminal.
#   2. Run this script. It asks for an experiment name + optional notes.
#   3. It snapshots the config (source YAML + the node's LIVE params) into a
#      timestamped folder, then records the diagnostics bag.
#   4. Press Ctrl-C to stop. It closes the bag and prints + saves a safety
#      report (collisions, clearances, keep-out, goal).
#
# Each run lands in:   $BAG_ROOT/<timestamp>_<name>/
#       bag/                 the rosbag2 recording
#       safe_ctrl.yaml       copy of the source config at record time
#       params_live.yaml     the node's actual loaded params (if node was up)
#       meta.txt             name, timestamp, notes, topics, host
#       report.txt           collision / clearance / goal analysis
#
# Override the bag root with:   BAG_ROOT=/some/dir ./record_experiment.sh
#
set -eo pipefail

# ---- locate the package (this script lives in <pkg>/scripts/) --------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_SRC="$PKG_DIR/config/safe_ctrl.yaml"
ANALYZER="$PKG_DIR/nick_safe_ctrl/run_analysis.py"
PLOTTER="$PKG_DIR/nick_safe_ctrl/plot_run.py"

# ---- config (override via env) ---------------------------------------------
BAG_ROOT="${BAG_ROOT:-$HOME/bags}"
NODE="${NODE:-/safe_ctrl_node}"
# Topics to record. Diagnostics alone is enough for analysis and is portable
# (decodes without the unitree msgs). Add the raw robot topics if you want them:
#   TOPICS=(/safe_ctrl_node/diagnostics /lf/sportmodestate /api/sport/request)
TOPICS=(/safe_ctrl_node/diagnostics)

# ---- ROS environment -------------------------------------------------------
source /opt/ros/foxy/setup.bash
[ -f /home/unitree/ros2_ws/install/setup.bash ] && source /home/unitree/ros2_ws/install/setup.bash
[ -f /home/unitree/nmr-cbf_ws/install/setup.bash ] && source /home/unitree/nmr-cbf_ws/install/setup.bash

# ---- prompt ----------------------------------------------------------------
read -rp "Experiment name: " NAME
[ -z "$NAME" ] && NAME="run"
read -rp "Notes (optional): " NOTES

STAMP="$(date +%Y%m%d_%H%M%S)"
SLUG="$(echo "$NAME" | tr ' /' '__' | tr -cd '[:alnum:]_-')"
DIR="$BAG_ROOT/${STAMP}_${SLUG}"
mkdir -p "$DIR"

# ---- snapshot the config ---------------------------------------------------
if [ -f "$CONFIG_SRC" ]; then
    cp "$CONFIG_SRC" "$DIR/safe_ctrl.yaml"
    echo "snapshot: source config -> safe_ctrl.yaml"
else
    echo "warn: source config not found at $CONFIG_SRC"
fi

if ros2 node list 2>/dev/null | grep -qx "$NODE"; then
    if ros2 param dump "$NODE" --print > "$DIR/params_live.yaml" 2>/dev/null; then
        echo "snapshot: live node params -> params_live.yaml"
    fi
else
    echo "warn: $NODE is not running -- recording anyway, no live param snapshot."
    echo "      (start the node first if you want params_live.yaml.)"
fi

{
    echo "name: $NAME"
    echo "timestamp: $STAMP"
    echo "notes: $NOTES"
    echo "topics: ${TOPICS[*]}"
    echo "host: $(hostname)"
} > "$DIR/meta.txt"

# ---- record (Ctrl-C to stop) -----------------------------------------------
echo
echo "Recording ${TOPICS[*]}"
echo "  -> $DIR/bag"
echo "Press Ctrl-C to stop."
echo
# A trap (not SIG_IGN) lets ros2 bag record receive SIGINT and shut the bag
# down cleanly, while keeping THIS script alive to run the analysis afterward.
trap 'printf "\n  stopping recording...\n"' INT
ros2 bag record -o "$DIR/bag" "${TOPICS[@]}" || true
trap - INT

# ---- analyze ---------------------------------------------------------------
echo
if [ -f "$ANALYZER" ]; then
    # tee the report to report.txt; don't let a non-zero (collision) exit abort us
    python3 "$ANALYZER" "$DIR/bag" --json "$DIR/report.json" | tee "$DIR/report.txt" || true
else
    echo "warn: analyzer not found at $ANALYZER -- skipping report."
fi

# ---- plot ------------------------------------------------------------------
echo
if [ -f "$PLOTTER" ]; then
    if python3 "$PLOTTER" "$DIR/bag" --labels "$SLUG" --out "$DIR/plot.png"; then
        :  # plot_run.py prints "saved <path>" itself
    else
        echo "warn: plotting failed (matplotlib missing?) -- bag + report are still saved."
    fi
else
    echo "warn: plotter not found at $PLOTTER -- skipping PNG."
fi

echo
echo "saved experiment to: $DIR"
