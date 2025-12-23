#!/bin/bash

# Monitor alignment agent sessions
SESSIONS=(
    "align-black-fab-compute"
    "align-initial-compute"
    "align-initial-energy"
    "align-fab-energy"
    "align-survival-rate"
    "align-likelihood-ratio"
)

CHECK_INTERVAL=60  # seconds

while true; do
    echo ""
    echo "=========================================="
    echo "ALIGNMENT AGENT STATUS CHECK - $(date)"
    echo "=========================================="

    for session in "${SESSIONS[@]}"; do
        echo ""
        echo "--- $session ---"
        # Capture last 20 lines of the session
        tmux capture-pane -t "$session" -p -S -20 2>/dev/null | tail -15
        echo ""
    done

    echo "=========================================="
    echo "Next check in $CHECK_INTERVAL seconds..."
    echo "Press Ctrl+C to stop monitoring"
    echo "=========================================="

    sleep $CHECK_INTERVAL
done
