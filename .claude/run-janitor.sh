#!/bin/bash

# Janitor Agent Runner Script
# Runs the janitor agent in a tmux session you can attach to

PROJECT_DIR="/Users/joshuaclymer/github/ai_futures_simulator"

# Send macOS notification that janitor is starting
osascript -e 'display notification "Janitor starting - run: tmux attach -t janitor" with title "Janitor Agent" sound name "Ping"'

# Create tmux session if it doesn't exist, then send the claude command
if ! tmux has-session -t janitor 2>/dev/null; then
  tmux new-session -d -s janitor -c "$PROJECT_DIR"
fi

# Send the janitor command to the tmux session
JANITOR_PROMPT="You are the janitor agent. Your job: (1) Do a thorough sweep of the codebase and add code cleanliness ideas to code_cleanliness_status.md (2) Get confirmation before making any actual code changes (3) When making changes, avoid changing functionality - only change structure, organization, naming, etc."

tmux send-keys -t janitor "cd $PROJECT_DIR && claude --resume janitor --append-system-prompt '$JANITOR_PROMPT' 'Activate as the janitor agent. Do a thorough sweep of the codebase for cleanliness issues. Update code_cleanliness_status.md with your findings, then present a summary and wait for my approval before making any changes.'" Enter
