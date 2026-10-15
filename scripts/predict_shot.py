"""Run a trained model over one shot as if live: alarm time, warning before the quench, and a plot."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import config, labels, notify, predict, registry
from src.data import load_shot


def run(shot_id, tag=None, telegram=False):
    """Predict one shot with the saved model `tag` (default: best saved). Returns a one-paragraph summary."""
    tag = tag or registry.best_tag()
    if tag is None:
        return "no trained model saved yet: run scripts/run_all.py first"
    model, meta = registry.load(tag)
    shot = load_shot(shot_id)
    win, t_alarm = predict.predict(shot, model, meta)
    t_quench = labels.detect_disruption(shot["time"], shot["ip"])["t_disrupt"]
    out_dir = os.path.join(config.RESULTS_DIR, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"shot_{shot_id}_{tag}.png")
    predict.plot_prediction(shot, win, t_alarm, meta["threshold"], t_quench, f"shot {shot_id}, model {tag}", path)
    text = f"shot {shot_id}, model {tag} ({meta['model']}): {len(win)} windows; "
    text += f"alarm at {t_alarm:.3f} s" if np.isfinite(t_alarm) else "no alarm"
    if np.isfinite(t_quench):
        text += f"; detector quench at {t_quench:.3f} s"
        if np.isfinite(t_alarm):
            text += f" ({1000 * (t_quench - t_alarm):.0f} ms warning)"
    if telegram:
        notify.send_file(path, text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--model", help=f"saved model tag under {config.MODELS_DIR} (default: best)")
    parser.add_argument("--telegram", action="store_true", help="also send the plot to Telegram")
    args = parser.parse_args()
    print(run(args.shot, args.model, args.telegram))


if __name__ == "__main__":
    main()
