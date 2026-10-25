# How this project works, in simple words

Read this first. `CODE_MAP.md` has the details of every file; this page is the story.

## 1. The problem
A tokamak holds a hot plasma with magnets. Sometimes the plasma suddenly collapses: a **disruption**. In a big machine like ITER a disruption can damage the walls, so the machine must see it coming and act (slow down, or shut down gently). We want a program that watches the signals and raises an **alarm** early enough.

## 2. The data
MAST was a UK tokamak (2000-2013). Each experiment is a **shot**, a few tenths of a second long. The UKAEA put the data online (FAIR-MAST). For each shot we read:
- 4 basic signals, every millisecond: plasma current, radiated power, neutron rate, beam heating power;
- extra signals: magnetic coils that see "locked modes" (a known warning sign), fast magnetic wobbles (Mirnov coils), the programmed current and the solenoid current;
- the operator's note, e.g. "DISRUPTION AT 220MS".

## 3. One shot's journey through the code
1. **Download** (`src/data.py`, `src/extras.py`). If STFC blocks our server, GitHub downloads for us (`src/relay.py`).
2. **Label: did it disrupt, and when?** (`src/labels.py`). Two witnesses: the operator's note, and a detector that looks for a sudden drop of the current. We call a shot "disrupted" only when **both** agree. Why: the current drops suddenly at the end of most MAST shots, even normal ones (74% of shots the operators called fine).
3. **Cut into windows** (`src/features.py`, `src/dataset.py`). Every 5 ms we look at the last 20 ms and compute simple numbers (average, slope, ...). A window only looks **backwards**, never at the future. That is the most important rule: no cheating.
4. **Split by shot** (`src/dataset.py`). 60% of shots to learn from, 20% to tune, 20% kept for one final test. A shot is never in two groups.
5. **Train models** (`src/models.py`):
   - physics rule: no learning, a fixed rule;
   - logistic regression: one weighted sum;
   - random forest: many decision trees that vote;
   - gradient boosting: trees that fix each other's mistakes.
6. **Alarm and score** (`src/alarms.py`, `src/evaluate.py`). The model gives a danger score every 5 ms. Two high scores in a row = alarm. Then each shot is one of:
   - **caught** (warned at least 5 ms early);
   - **late**;
   - **missed**;
   - **false alarm** (an alarm on a normal shot);
   - **quiet** (a normal shot with no alarm).

   The alarm level is chosen so at most 5% of normal shots get a false alarm.
7. **Experiments** (`src/experiment.py`): the same pipeline with one thing changed each time.

| Exp | Question |
|---|---|
| E0-E3 | How good is each model? |
| E4 | How far ahead should it look? |
| E5 | Which signal matters? |
| E6 | Does it still work on later years (campaigns)? |
| E7 | Which labels to trust? |
| E8 | Can we *promise* a false-alarm rate (conformal prediction)? |
| E9 | What if the unnoted current drops are "unknown" instead of "normal" (PU learning)? |
| E10 | Repeat everything with 50 different random splits, to see how much is luck |
| E11 | Is the model cheating with "time since start" (shortcut test)? |
| E12 | The rules of the UKAEA benchmark |
| E13 | Do the magnetic signals help? |

## 4. The machine around it
- **Research server** (`scripts/research_server.py`): runs forever on the VPS, retrains when the code changes, runs E10 when idle, obeys only your Telegram chat.
- **Resume** (`src/resume.py`): every result has a fingerprint of the code and data that made it. Same fingerprint, so it is reused, not retrained.
- **Status picture** (`src/status_card.py`): the one picture Telegram sends. Details are in `results/REPORT.md` (`/report`), written by `src/explain.py`.
- **Watchers**: `src/sources.py` (is the data site back? did FAIR-MAST grow?) and `src/literature.py` (new papers on arXiv every week).
- **Automatic labels** (`scripts/auto_labels.py`): the server works out how each shot ended, from two signals at the moment the current dropped. Was the programme already ramping the current down? Had the solenoid (the coil that drives the current) stopped sweeping, meaning the machine ran out of flux? If neither, the plasma was lost while everything was still running: most likely a real disruption. No hand labelling needed.

## 5. What the results say so far (400 shots, real runs)
- With the 4 basic signals, the model catches about as many disruptions as a random alarm would (E10, 43 repeat runs). That is a real result: these signals alone are not enough on MAST.
- Radiated power is the signal it relies on most (E5).
- Trained on early years, the alarm level gives too many false alarms in later years (E6).
- Part of what it learned was "how long the shot has lasted", a shortcut, not physics (E11).
- False alarms fall mostly on "normal" shots that end in a sudden current drop nobody noted: some of those may be disruptions with a wrong label.

## 6. What is different here
What others have done (so it is **not** new here, we cite them):
- UKAEA (the owners of FAIR-MAST) trained a deep network (BiLSTM) on 417 MAST shots, with labels from the current drop alone.
- Groups working on other machines (DIII-D, JET, Alcator C-Mod, EAST; e.g. MIT and Princeton) built random forests and deep networks.
- Others have also used conformal prediction and time-to-disruption models.

What this project adds, which we **could not find** in published work (never say "nobody has done it", only "we could not find it"):
1. **A measured check of MAST's labels.** The current-drop detector also fires on 74% of shots the operators called normal. So labels from the current alone, like the UKAEA benchmark's, mix normal endings with disruptions.
2. **PU learning for unrecorded disruptions** (E9): unnoted current drops are treated as "unknown", not "normal". It also estimates how many of them look like real disruptions.
3. **Testing the false-alarm promise when the machine changes between years** (E6 + E8), on MAST.
4. **Honesty tools built in:**
   - the shortcut test (E11);
   - 50 repeat runs with error bars and a comparison with random guessing (E10);
   - where the false alarms fall;
   - automatic, physics-based termination labels that point to disruptions nobody recorded.
5. **A first look at MAST's saddle-coil signal** for locked modes, including a correction that did *not* work (NOTES.md, 2026-10-16). Negative results count in research.

## 7. How to study it (one step a day)
`src/config.py` -> `src/labels.py` -> `src/features.py` -> `src/dataset.py` -> `src/alarms.py` -> `src/models.py` -> `src/evaluate.py` -> `src/experiment.py` -> `scripts/run_all.py` -> `src/resume.py` -> `scripts/research_server.py`. For each file, explain back in your own words what goes in, what comes out, and why it exists.
