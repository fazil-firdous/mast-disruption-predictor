"""Save and load trained models together with what is needed to use them: feature columns and alarm threshold.

Models go to MODELS_DIR, which on the owner's server is outside the repo, so trained models stay private.
Loading uses pickle, which can run code hidden in a file: only load models you trained yourself.
"""
import json
import os
import pickle

from src import config


def save(model, tag, name, columns, threshold, summary):
    """Write model.pkl and meta.json to MODELS_DIR/tag and return that folder."""
    folder = os.path.join(config.MODELS_DIR, tag)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    meta = {"tag": tag, "model": name, "columns": list(columns), "threshold": float(threshold),
            "k_consec": config.K_CONSEC, "label_source": config.LABEL_SOURCE, "summary": summary}
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return folder


def load(tag):
    """(model, meta) saved under MODELS_DIR/tag."""
    folder = os.path.join(config.MODELS_DIR, tag)
    with open(os.path.join(folder, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(folder, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    return model, meta


def saved_tags():
    """Tags of every saved model, e.g. ['E1', 'E3']."""
    if not os.path.isdir(config.MODELS_DIR):
        return []
    return sorted(t for t in os.listdir(config.MODELS_DIR) if os.path.exists(os.path.join(config.MODELS_DIR, t, "meta.json")))


def best_tag():
    """The saved model with the highest validation catch rate, or None if nothing is saved."""
    best, best_rate = None, -1.0
    for tag in saved_tags():
        with open(os.path.join(config.MODELS_DIR, tag, "meta.json")) as f:
            rate = json.load(f)["summary"].get("val_catch_rate") or 0.0
        if rate > best_rate:
            best, best_rate = tag, rate
    return best
