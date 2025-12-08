#!/bin/bash

# Script to set up new git worktrees for the AI Progress Calculator project
# Usage: ./setup-worktree.sh <branch-name> [base-branch]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if branch name is provided
if [ -z "$1" ]; then
    print_error "Usage: $0 <branch-name> [base-branch]"
    print_error "Example: $0 feature/new-charts main"
    exit 1
fi

BRANCH_NAME="$1"
BASE_BRANCH="${2:-main}"
CURRENT_DIR=$(pwd)
PROJECT_NAME=$(basename "$CURRENT_DIR")
WORKTREE_DIR="../${PROJECT_NAME}-${BRANCH_NAME}"

# Validate we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Current directory is not a git repository"
    exit 1
fi

print_status "Setting up worktree for branch: $BRANCH_NAME"
print_status "Base branch: $BASE_BRANCH"
print_status "Worktree directory: $WORKTREE_DIR"

# Check if worktree directory already exists
if [ -d "$WORKTREE_DIR" ]; then
    print_error "Worktree directory already exists: $WORKTREE_DIR"
    exit 1
fi

# Fetch latest changes from remote
print_status "Fetching latest changes from lw remote..."
git fetch lw

# Check if base branch exists
if ! git show-ref --verify --quiet "refs/remotes/lw/$BASE_BRANCH"; then
    print_error "Base branch 'lw/$BASE_BRANCH' does not exist"
    exit 1
fi

# Create new worktree
print_status "Creating worktree..."
git worktree add -b "$BRANCH_NAME" "$WORKTREE_DIR" "lw/$BASE_BRANCH"

# Navigate to the new worktree
cd "$WORKTREE_DIR"

# Set up remote tracking
print_status "Setting up remote tracking..."
git branch --set-upstream-to=lw/main "$BRANCH_NAME"

# Configure git settings for this worktree
print_status "Configuring git settings..."
git config push.default simple
git config remote.pushDefault lw

# Create .claude directory if it doesn't exist
if [ ! -d ".claude" ]; then
    print_status "Creating .claude directory..."
    mkdir -p .claude
fi

# Copy .claude/settings.local.json from the original worktree
if [ -f "$CURRENT_DIR/.claude/settings.local.json" ]; then
    print_status "Copying Claude settings..."
    cp "$CURRENT_DIR/.claude/settings.local.json" ".claude/settings.local.json"
    print_success "Claude settings synchronized"
else
    print_warning "No Claude settings found in original worktree"
fi

# Copy CLAUDE.md if it exists
if [ -f "$CURRENT_DIR/CLAUDE.md" ]; then
    print_status "CLAUDE.md will be available from git history"
else
    print_warning "No CLAUDE.md found in original worktree"
fi

# Install dependencies if package.json exists
if [ -f "package.json" ]; then
    print_status "Installing Node.js dependencies..."
    npm install
    print_success "Dependencies installed"
fi

# Install Python dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    print_status "Python dependencies will be installed on first 'npm run dev'"
    print_warning "Run 'npm run dev' to install Python dependencies and start development"
fi

print_success "Worktree setup complete!"
print_status "Location: $WORKTREE_DIR"
print_status "Branch: $BRANCH_NAME (tracking lw/main)"
print_status ""
print_status "To start working:"
print_status "  cd $WORKTREE_DIR"
print_status "  npm run dev"
print_status ""
print_status "To remove this worktree later:"
print_status "  git worktree remove $WORKTREE_DIR"

# Return to original directory
cd "$CURRENT_DIR"