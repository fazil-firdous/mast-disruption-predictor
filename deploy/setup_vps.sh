#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu 22.04/24.04 VPS (re-running it is safe). From the cloned repo:
#   bash deploy/setup_vps.sh
# Public code, private runs: the repo stays a clean copy of GitHub; downloaded data, trained models, results
# and logs go to MAST_HOME (default ~/mast-private), and the bot token stays in the git-ignored .env.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$APP_DIR"

sudo apt-get update -y
sudo apt-get install -y git python3 python3-venv python3-pip screen

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

touch .env
chmod 600 .env
if ! grep -q '^TELEGRAM_BOT_TOKEN=' .env; then
  read -rp "Telegram bot token: " token
  echo "TELEGRAM_BOT_TOKEN=$token" >> .env
fi
if ! grep -q '^TELEGRAM_CHAT_ID=' .env; then
  read -rp "Telegram chat id: " chat
  echo "TELEGRAM_CHAT_ID=$chat" >> .env
fi
if ! grep -q '^MAST_HOME=' .env; then
  echo "MAST_HOME=$HOME/mast-private" >> .env
fi
mkdir -p "$(grep '^MAST_HOME=' .env | cut -d= -f2)"

# the old nightly cron job is replaced by the research server
( crontab -l 2>/dev/null | grep -v run_pipeline.sh ) | crontab - || true

.venv/bin/python -m pytest -q
.venv/bin/python -c "from src import notify; notify.send('VPS setup finished, bot works')"
.venv/bin/python scripts/diagnose_network.py || echo "Network check failed: read the Telegram report."

# a systemd service restarts the server after crashes and reboots
sudo tee /etc/systemd/system/mast-research.service > /dev/null <<UNIT
[Unit]
Description=MAST disruption research server
After=network-online.target
Wants=network-online.target

[Service]
User=$(id -un)
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/.venv/bin/python scripts/research_server.py --scale full
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now mast-research.service
echo "Done. The research server is running as a service. Send /help to your bot."
echo "Logs: journalctl -u mast-research -f    Stop: sudo systemctl stop mast-research"
