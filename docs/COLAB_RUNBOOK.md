# Colab Training Runbook

**Why this exists:** local CPU fine-tuning (`scripts/train_local.py`) does not fit on this VM
alongside the live FatBaby pipeline. Confirmed 2026-07-17: with signalapi/processor/entity-graph/
newssite/secwatch running (~3GB combined RSS on a 3.8GB box), free RAM drops to ~140-300MB the
moment `train_local.py` starts loading the model, and the process is silently killed (no
traceback — consistent with an OOM kill, not a script bug) within seconds of starting step 0,
twice, at two different memory footprints. **Colab's free-tier T4 GPU is the real path** — this
was already Milestone 2's design (`notebooks/gpt2_finetune_colab.ipynb` has existed since
2026-06-14); this doc is the concrete step-by-step to actually run it.

Corpus is already built and current as of this runbook: `var/emily-corpus.jsonl`, 1048 records,
1.3MB (`python3 scripts/prime_directive_dataset.py --emily-root /home/fatbaby/EMILY --apples-dir
/home/fatbaby/APPLES --output var/emily-corpus.jsonl --colab`, run 2026-07-17, reflects the full
current golden-docs-index including today's NORN/PRIME-097/TYLER additions). Re-run that command
before starting if it's been more than a day or two — golden docs change fast in this repo.

---

## 1. Get the corpus into Colab

`scripts/drive_sync.py` (the automated upload path) needs `GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON` set
on the IDUNA side — **not currently configured** (checked 2026-07-17, absent from
`IDUNA/var/*.env` and no running process has it set). Two options:

**Option A — manual upload (works right now, no setup):**
1. Download `var/emily-corpus.jsonl` from this machine to your local computer (`scp` or however
   you normally pull files off this VM).
2. Go to [drive.google.com](https://drive.google.com), create/open a folder named exactly
   `emily-training` at the top level of My Drive — this matches the notebook's default
   `DRIVE_FOLDER` (cell 3: `/content/drive/MyDrive/emily-training`), so you don't have to edit
   that cell. Drag the corpus file in.
3. If you'd rather use a different folder name, that's fine too — just edit `DRIVE_FOLDER` in
   cell 3 to match before running it.

**Option B — configure Drive API first (better for repeat runs):**
1. Create a GCP service account with Drive API access, download its JSON key.
2. On this machine: `export GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON=/path/to/key.json` and
   `export GOOGLE_DRIVE_FOLDER_ID=<folder-id>` in IDUNA's environment, restart IDUNA.
3. Then: `IDUNA_BASE_URL=http://localhost:8080 IDUNA_AGENT_NAME=EMILY-TRAINING
   IDUNA_AGENT_SECRET=<from IDUNA/var/agent-secrets.env, IDUNA_SECRET_EMILY_TRAINING>
   python3 scripts/drive_sync.py --upload var/emily-corpus.jsonl` (run from
   `/home/fatbaby/gpt2-alpine-c`).
4. Every future corpus rebuild is then one command instead of a manual download/upload.

Option A is faster for a one-off run today; do Option B if this becomes a repeated cycle.

## 2. Open the notebook and run the one cell (rewritten 2026-07-17)

The notebook is down to a single reusable bootstrap cell — this is the "paste once, hit play"
workflow, not a per-run walkthrough:

1. Open directly from GitHub: Colab → File → Open notebook → GitHub tab →
   `emilyspringerton/gpt2-alpine-c` → `notebooks/gpt2_finetune_colab.ipynb`. (Or upload it once —
   either way, you never edit this file's cells again.)
2. **Runtime → Change runtime type → T4 GPU** (free tier).
3. Run the one code cell. It mounts Drive (accept the OAuth prompt when it appears), clones
   `gpt2-alpine-c` into `/content/gpt2-alpine-c` (or `git pull`s it if already present from an
   earlier run in the same session), then executes `scripts/colab_train.py` — which installs
   `transformers`/`datasets`/`accelerate`, loads the corpus from
   `DRIVE_FOLDER/emily-corpus.jsonl`, tokenizes, trains, saves the checkpoint back to Drive,
   evaluates perplexity, and runs a generation smoke test — end to end, unattended.
4. If your Drive folder isn't named `emily-training` (the default from step 1's Option A), set
   `DRIVE_FOLDER` as an environment variable in the cell before the `subprocess.run` call, or pass
   `--drive-folder` by editing the final line — this is the *only* line in the whole workflow that
   should ever need a per-user tweak.

**Why this matters going forward:** all the actual training logic (tokenization, hyperparameters,
`TrainingArguments`, eval, generation prompts) lives in `scripts/colab_train.py`, in git — not
pasted into notebook cells. Change training behavior by editing and pushing that script; the next
"Run all" in Colab picks up the new version automatically via `git pull`, with nothing to
re-paste or manually resync. `scripts/colab_train.py`'s CLI flags (`--epochs`, `--batch-size`,
`--learning-rate`, etc., all env-var-overridable too) cover the tuning knobs the old per-cell
`Cell 3`/`Cell 6` config used to expose.

## 3. Bring the checkpoint back

1. From Drive (or Colab's file browser, left sidebar), download the output directory
   (`checkpoint-final/`, per `colab_train.py`'s `OUTPUT_DIR` default) as a `.tar.gz` — the script
   already archives it there for you — or download the individual files (`config.json`,
   `model.safetensors` or `pytorch_model.bin`, `tokenizer.json`, etc.).
2. Get it onto this machine into `gpt2-alpine-c/checkpoint-emily-ft/` (same path
   `train_local.py` already writes to, so downstream tooling doesn't need new flags) — `scp`,
   or `drive_sync.py --download --pattern "checkpoint*.tar.gz"` if you went with Option B above.

## 4. Convert and validate (same steps as any fine-tune, Colab or local)

```bash
cd /home/fatbaby/gpt2-alpine-c

# HuggingFace checkpoint → C binary weights
python3 scripts/convert_ft_checkpoint.py \
  --checkpoint ./checkpoint-emily-ft \
  --output ./weights/emily-ft.bin

# Perplexity: base vs fine-tuned, same eval settings as every prior milestone
python3 scripts/eval_perplexity.py \
  --corpus var/emily-corpus.jsonl \
  --checkpoint ./checkpoint-emily-ft \
  --compare --memory-efficient

# Entropy source check (what the RSI loop actually consumes)
./gpt2_run weights/emily-ft.bin --entropy-stats
```

**Target, per NORTHSTAR.md Milestone 3:** entropy delta ≥ 0.5 nats over base (base H_mean=4.4877).
The 2026-06-23 local 300-step CPU run only reached +0.17 nats — explicitly noted then as
"Colab T4 full fine-tune required" to hit target. This run is that attempt.

## 5. Commit the result

Checkpoints are large — this repo already has Git LFS set up for exactly this
(`emily-ft.bin`, `model.bin`, `checkpoint-emily-ft/model.safetensors` are LFS-tracked, per
CHANGELOG 2026-06-16). Standard flow:

```bash
git add weights/emily-ft.bin checkpoint-emily-ft/
git commit -m "feat(training): Colab T4 fine-tune — <N> epochs, <entropy delta> nats"
git push
```

Then update `NORTHSTAR.md` Milestone 2/3 status and file an Apple
(`emily apples post -t completion "Colab T4 fine-tune complete" "..."`) — matches the pattern
every prior training milestone in this repo's CHANGELOG.md already follows.

---

## If Colab's free tier isn't enough

Corpus is small (1048 records / ~1.3MB) — a full fine-tune should comfortably finish in one
free-tier T4 session. If you hit a usage-limit wall anyway: reduce `num_train_epochs` in cell 6,
or fall back to Colab Pro for guaranteed GPU time. Not expected to be necessary at this corpus
size.
