"""Send progress messages and result files to Telegram. Prints instead if no bot is configured."""
import os

import requests

from src import config  # noqa: F401  (importing config loads the private .env)

API = "https://api.telegram.org/bot{token}/{method}"


def _credentials():
    """(token, chat_id) from the environment (filled from .env by config), or (None, None)."""
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


def get_updates(offset, timeout=25):
    """New bot messages since offset, as a list of (update_id, chat_id, text). Empty if not configured or offline.

    Long polling: Telegram holds the request open for up to timeout seconds until a message arrives.
    """
    token, _ = _credentials()
    if not token:
        return []
    try:
        response = requests.get(API.format(token=token, method="getUpdates"),
                                params={"offset": offset, "timeout": timeout}, timeout=timeout + 10)
        return parse_updates(response.json())
    except (requests.RequestException, ValueError) as error:
        print(f"[notify] getUpdates failed: {error}")
        return []


def parse_updates(payload):
    """Turn a getUpdates JSON reply into (update_id, chat_id, text) tuples, skipping non-text updates."""
    out = []
    for update in payload.get("result", []):
        message = update.get("message") or {}
        text = message.get("text")
        if text:
            out.append((update["update_id"], str(message.get("chat", {}).get("id")), text))
        else:
            out.append((update["update_id"], None, None))
    return out


def owner_chat_id():
    """The only chat whose commands the research server obeys."""
    return _credentials()[1]
