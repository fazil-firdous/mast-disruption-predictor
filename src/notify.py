"""Telegram messages and pictures for the owner, and a log of everything (LOGS_DIR/messages.log). Prints instead
of sending if no bot is configured."""
import os
import time

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


def remember(text):
    """Append a message to LOGS_DIR/messages.log with the time, so everything stays on record. Never raises."""
    try:
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        with open(os.path.join(config.LOGS_DIR, "messages.log"), "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")
    except OSError as error:
        print(f"[notify] could not write messages.log: {error}")


def log(text):
    """Keep a message off Telegram: print it (it lands in the job's log) and keep it in messages.log."""
    print(f"[log] {text}")
    remember(text)


def send(text):
    """Send a text message (Telegram's limit is 4096 characters, so longer text is cut); also kept in messages.log."""
    print(f"[notify] {text}")
    remember(text)
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
