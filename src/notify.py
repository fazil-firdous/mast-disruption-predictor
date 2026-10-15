"""Send progress messages and result files to Telegram. Prints instead if no bot is configured."""
import os

import requests

API = "https://api.telegram.org/bot{token}/{method}"
ENV_FILE = ".env"


def _load_env_file(path=ENV_FILE):
    """Read KEY=VALUE lines from .env into os.environ (values already set in the environment win)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"'))


def _credentials():
    """(token, chat_id) from the environment or .env, or (None, None)."""
    _load_env_file()
    return os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")


def _call(method, data, files=None):
    """POST one Bot API call. Never raises: a failed notification must not stop a training run."""
    token, chat_id = _credentials()
    if not token or not chat_id:
        return False
    try:
        response = requests.post(API.format(token=token, method=method), data={"chat_id": chat_id, **data},
                                 files=files, timeout=30)
        return response.ok
    except requests.RequestException as error:
        print(f"[notify] telegram failed: {error}")
        return False


def send(text):
    """Send a text message (Telegram's limit is 4096 characters, so longer text is cut)."""
    print(f"[notify] {text}")
    return _call("sendMessage", {"text": text[:4000]})


def send_file(path, caption=""):
    """Send a PNG as a photo, anything else as a document. Missing files are skipped."""
    if not os.path.exists(path):
        return False
    method, field = ("sendPhoto", "photo") if path.endswith(".png") else ("sendDocument", "document")
    with open(path, "rb") as f:
        return _call(method, {"caption": caption[:1000]}, files={field: f})
