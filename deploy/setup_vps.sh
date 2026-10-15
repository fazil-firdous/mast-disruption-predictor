#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu 22.04/24.04 VPS (re-running it is safe). From the cloned repo:
#   bash deploy/setup_vps.sh
# Public code, private runs: the repo stays a clean copy of GitHub. Downloaded data, trained models, results, logs
# and the private settings (~/mast-private/.env: tokens, chat id) live in ~/mast-private, which survives deleting
# or re-cloning the code folder.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$APP_DIR"

sudo apt-get update -y
sudo apt-get install -y git python3 python3-venv python3-pip screen

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

PRIVATE="$HOME/mast-private"
ENV_FILE="$PRIVATE/.env"
mkdir -p "$PRIVATE"
touch "$ENV_FILE"
chmod 600 "$ENV_FILE"

# settings from an older setup lived in the repo's .env; copy any missing keys to the permanent folder
if [ -f .env ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    key="${line%%=*}"
    if [ -n "$key" ] && [ "${key:0:1}" != "#" ] && ! grep -q "^$key=" "$ENV_FILE"; then
      echo "$line" >> "$ENV_FILE"
    fi
  done < .env
  mv .env "$PRIVATE/.env.from-repo.bak"
fi

ask() {
  if ! grep -q "^$1=" "$ENV_FILE"; then
    read -rp "$2" value
    if [ -n "$value" ]; then echo "$1=$value" >> "$ENV_FILE"; fi
  fi
}
ask TELEGRAM_BOT_TOKEN "Telegram bot token: "
ask TELEGRAM_CHAT_ID "Telegram chat id: "
ask GITHUB_TOKEN "GitHub fine-grained token for the data relay (Enter to skip): "
grep -q '^MAST_HOME=' "$ENV_FILE" || echo "MAST_HOME=$PRIVATE" >> "$ENV_FILE"
repo="$(git remote get-url origin | sed -E 's#^(git@github.com:|https://github.com/)##; s#\.git$##')"
grep -q '^GITHUB_REPO=' "$ENV_FILE" || echo "GITHUB_REPO=$repo" >> "$ENV_FILE"

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
sudo systemctl enable mast-research.service
sudo systemctl restart mast-research.service
echo "Done. The research server is running as a service. Send /help to your bot."
echo "Logs: journalctl -u mast-research -f    Stop: sudo systemctl stop mast-research"
