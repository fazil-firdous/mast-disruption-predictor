#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu 22.04/24.04 VPS. Run as a normal user with sudo:
#   curl -fsSL https://raw.githubusercontent.com/fazil-mastu/mastu/main/deploy/setup_vps.sh | bash
# (or clone the repo first and run: bash deploy/setup_vps.sh)
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/fazil-mastu/mastu.git}"
BRANCH="${BRANCH:-main}"
APP_DIR="$HOME/mastu"

sudo apt-get update -y
sudo apt-get install -y git python3 python3-venv python3-pip tmux

if [ ! -d "$APP_DIR/.git" ]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# the bot token lives only on this machine, never in git (.env is in .gitignore)
if [ ! -f .env ]; then
  read -rp "Telegram bot token: " token
  read -rp "Telegram chat id: " chat
  printf 'TELEGRAM_BOT_TOKEN=%s\nTELEGRAM_CHAT_ID=%s\n' "$token" "$chat" > .env
  chmod 600 .env
fi

.venv/bin/python -c "from src import notify; notify.send('VPS setup finished, bot works')"
.venv/bin/python -m pytest -q

# re-run every night at 03:10 if the code changed (run_pipeline.sh skips otherwise)
( crontab -l 2>/dev/null | grep -v run_pipeline.sh ; echo "10 3 * * * $APP_DIR/deploy/run_pipeline.sh >> $APP_DIR/logs/cron.log 2>&1" ) | crontab -
mkdir -p logs
echo "Setup done. Start the first run with:  tmux new -s mast '$APP_DIR/deploy/run_pipeline.sh --force'"
