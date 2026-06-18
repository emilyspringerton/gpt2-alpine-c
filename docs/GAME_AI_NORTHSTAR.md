# Game AI Northstar — GPT-2 as a Game Policy Network

*Written: 2026-06-18*

---

## The Three-Sentence Version

GPT-2's token generation maps naturally onto sequential game decision-making: encode game state
as a token sequence, generate action tokens, decode to game inputs. The testbed is SHANKPIT
(our own game — full source control, instant iteration), with BedWars as the MOBA-scale target
that mirrors the strategic complexity of League of Legends. The end state is a self-improving
game AI that learns from replay data, generates its own training data via self-play, and runs
in real-time on CPU via the existing C inference engine.

---

## Why GPT-2 for Game AI

GPT-2 is not the obvious choice — but it has concrete advantages for this use case:

1. **Pre-trained spatial language understanding.** GPT-2 already knows "north of", "above",
   "approaching at range", "flanking from the left". A game state token like `"enemy 12m northeast
   health 60%"` activates semantic representations the model already has. A randomly-initialized
   policy network has to learn these from scratch.

2. **Compositional generalization.** Describe a novel tactic in game-state tokens the model has
   never seen together — GPT-2 can generalize from components. A ResNet policy cannot.

3. **Interpretable intermediate tokens.** The model's chain of reasoning is visible as tokens.
   You can inspect what the model "noticed" before deciding to shoot.

4. **Fine-tune efficiency.** 466 records trained the Emily Prime model in 2.2 minutes on Colab T4.
   A game replay corpus of 10K episodes trains in under an hour.

5. **C inference at game speed.** The existing `gpt2_run` C binary generates tokens at ~200 tok/s
   on a laptop CPU. At 50 state tokens per tick → action decision in ~250ms. That's real-time
   enough for a 4 Hz decision loop (faster than most human reaction windows in strategic decisions).

---

## Architecture

```
Game Loop (SHANKPIT server tick)
          │
          ▼
  ┌──────────────────┐
  │  State Serializer │  ← game_state → natural language token sequence
  └────────┬─────────┘    "player pos:14,8 hp:85% weapon:ar ammo:22
           │                enemy at:20,15 dist:12 hp:60% visible
           │                flag:carried objective:home_base"
           ▼
  ┌──────────────────┐
  │   GPT-2 Policy   │  ← fine-tuned on (state → action) replay pairs
  └────────┬─────────┘    next tokens: "move:northeast strafe:0.2
           │                aim:20,15 shoot:1 weapon:ar"
           ▼
  ┌──────────────────┐
  │  Action Decoder  │  ← token sequence → UserCmd (fwd, str, yaw, buttons)
  └────────┬─────────┘
           ▼
    PacketUserCmd → SHANKPIT server
```

The inference server (`scripts/serve.py` on :8088) is already wired. The state serializer
and action decoder are the new pieces.

---

## Game State Token Format

Natural language, not numeric. GPT-2 was pre-trained on language — use that.

### SHANKPIT state (50-80 tokens)

```
shankpit state tick:12450 scene:0 mode:ctf
self pos:14,8 hp:85 shield:30 weapon:ar ammo:22 crouch:0
team: pos:8,12 hp:100 | pos:22,3 hp:60
enemy: pos:20,15 hp:60 vis:1 | pos:5,4 hp:? vis:0
flag red:home blue:self
```

### BedWars state (80-120 tokens)

```
bedwars state tick:8820 team:blue
self pos:8,5,-6 hp:100 iron:24 gold:3 sword:iron armor:chain
bed blue:intact red:intact green:destroyed yellow:intact
enemy red: pos:88,5,2 hp:80 vis:1 | pos:85,5,6 hp:100 vis:0
gen center:diamond 18s iron:2s gold:8s
shop npc:dist:4
```

### Action token format (10-20 tokens)

```
move:northeast fwd:0.8 str:0.2 yaw_delta:-2.3 shoot:1 jump:0 crouch:0 weapon:ar
```

The token vocabulary is small enough (~200 domain tokens) that fine-tuning the GPT-2
embedding layer on game-specific tokens is cheap and fast.

---

## Milestones

### Milestone 6: SHANKPIT State Serializer + Action Decoder

**What:** Python module `scripts/game_state.py` with two functions:
- `serialize_snapshot(snapshot_bytes) → str` — parse PacketSnapshot, format as game state string
- `decode_action(action_str) → UserCmd` — parse GPT-2 output tokens into UserCmd struct

**Why first:** The state/action format is the contract between all other milestones.
Everything downstream depends on getting this right.

**Acceptance:**
- `serialize_snapshot(raw_packet)` returns a stable string for a fixed input
- `decode_action("move:northeast fwd:0.8 shoot:1")` returns a valid UserCmd dict
- Round-trip: encode → decode → re-encode is stable
- Python unit tests pass

**Output files:**
- `scripts/game_state.py` — serializer + decoder
- `tests/test_game_state.py` — unit tests with fixture packets

---

### Milestone 7: Replay Logger in emily-bot

**What:** `apps2/emily-bot/main.go` logs `(state_str, action_str)` pairs to a replay NDJSON file.
Each line: `{"tick": 12450, "state": "...", "action": "..."}`.

After every bot session, replay data accumulates at `var/replays/YYYYMMDD-HHmm.ndjson`.

**Why:** No fine-tuning data → no game AI. This milestone makes the data pipeline automatic
— every session the bots play generates training data.

**Acceptance:**
- Bot session produces `var/replays/*.ndjson` with ≥1 record per tick
- Each record has `state` (matches serialize format) and `action` (matches action format)
- 100-tick session produces ~100 records
- Replay file readable by `scripts/build_game_corpus.py`

**Output files:**
- `apps2/emily-bot/main.go` (modified) — add state serialization + replay logging
- `scripts/build_game_corpus.py` — aggregates replay NDJSON → training JSONL

---

### Milestone 8: Fine-Tune GPT-2 on Replay Corpus

**What:** Extend `prime_directive_dataset.py` to accept a `--game-replays <dir>` flag that
ingests SHANKPIT replay NDJSON files. Each record becomes an instruction pair:

```json
{"prompt": "shankpit state tick:12450 scene:0 ...", "completion": "move:northeast shoot:1 weapon:ar"}
```

Combine with Emily corpus (--colab preset) for a mixed fine-tune:
30% game pairs / 70% Emily operational text.

Fine-tune on Colab T4 with the existing `gpt2_finetune_colab.ipynb` notebook.

**Acceptance:**
- `--game-replays var/replays/` adds game pairs to the JSONL corpus
- Colab fine-tune produces a checkpoint with game loss < 2.0 (vs random ~6.4)
- `gpt2_run` running the game checkpoint generates valid action tokens when prompted with a game state string

---

### Milestone 9: GPT-2 Policy in emily-bot

**What:** Replace emily-bot's heuristic `think()` with a call to the GPT-2 inference server
(`POST http://localhost:8088/generate`). At each 4 Hz tick:
1. Serialize current game state → prompt string
2. POST to inference server → action string
3. Decode action string → UserCmd
4. Send UserCmd to SHANKPIT server

The heuristic stays as a fallback when the inference server is unavailable.

**Acceptance:**
- `emily-bot -gpt2-url http://localhost:8088` activates the GPT-2 policy
- Bot connects, plays a session, generates valid UserCmds without crashing
- Action distribution is non-trivial (bot doesn't just stand still or spam shoot)

---

### Milestone 10: Self-Play Loop

**What:** Automated loop:
1. `emily start --shankpit --bots 4` — 4 GPT-2 bots play a session
2. Replay data written to `var/replays/`
3. `scripts/build_game_corpus.py` appends new replay records
4. Colab re-fine-tune (triggered manually or via Drive sync)
5. New checkpoint deployed → bots play again

This is the flywheel. Each generation of bots is trained on data from the previous generation.

**Acceptance:**
- Manual trigger of one full loop produces a second-generation checkpoint
- Second-gen bots measurably different from first-gen (action distribution shifts)

---

### Milestone 11: BedWars AI (MOBA-Scale Northstar)

**What:** Extend the state serializer to BedWars state format (beds, generators, shop,
resource inventory). Fine-tune on BedWars replays. The bot understands:
- When to rush the enemy bed vs. farm generators
- When to buy armor vs. weapons
- When to defend vs. push

This is the "our own League of Legends" milestone. BedWars has the same strategic primitives
as a MOBA: economy (gold/resources → items), objectives (beds/nexus), team elimination,
map control (island generators). The GPT-2 bot operating in BedWars at this level is the
proof-of-concept that the architecture scales to MOBA complexity.

**Why this is the northstar, not actual LoL:**
- Full source control — we can log any state we want
- No ToS issues
- Faster iteration (no reverse engineering the game client)
- BedWars → LoL is a vocabulary extension, not an architecture change
- The architecture paper writes itself: "GPT-2 plays competitive BedWars via token-serialized game state"

**Acceptance:**
- BedWars GPT-2 bot completes a full game (not just connects)
- Bot makes at least one economically correct decision (buys gear, mines resources, targets beds)
- Replay quality measurably improves from generation 1 → generation 3 self-play

---

## Token Vocabulary Design

A small lexicon of ~200 domain tokens is sufficient. GPT-2's BPE tokenizer handles unknown
words by splitting them — but pre-defined vocabulary tokens are more stable.

| Category | Examples |
|----------|---------|
| Scene primitives | `pos:`, `hp:`, `dist:`, `vis:` |
| Directions | `north`, `northeast`, `east`, `southeast`, `south`, `southwest`, `west`, `northwest` |
| Actions | `move:`, `shoot:`, `jump:`, `crouch:`, `reload:`, `use:` |
| Weapons | `knife`, `magnum`, `ar`, `shotgun`, `sniper`, `katana` |
| Game modes | `shankpit`, `bedwars`, `ctf`, `tdm` |
| BedWars domain | `bed`, `intact`, `destroyed`, `iron:`, `gold:`, `diamond:`, `gen`, `shop`, `armor` |
| Team markers | `team:blue`, `team:red`, `ally:`, `enemy:` |
| Objectives | `flag:`, `base:`, `home:`, `captured` |

These tokens already exist in GPT-2's vocabulary (they're common English words). The
fine-tune teaches the model the game-domain semantic associations.

---

## Connection to SHANKPIT BedWars

Milestone 11 (BedWars AI) is the convergence point between this repo and the SHANKPIT
BedWars spec (`SHANKPIT/docs2/specs/BEDWARS_SPEC.md`). The two milestones are parallel:

| Track | What's being built |
|-------|--------------------|
| SHANKPIT Milestone 7 | BedWars server + client (the game) |
| GPT-2 Milestone 11 | BedWars AI bot (the player) |

They can be developed concurrently: SHANKPIT builds the game, gpt2-alpine-c builds the
player that learns to play it.

---

## Related Specs

- `NORTHSTAR.md` — parent northstar (Emily Prime inference track, Milestones 0-5)
- `SHANKPIT/docs2/specs/BEDWARS_SPEC.md` — BedWars game mode (the training environment)
- `SHANKPIT/apps2/emily-bot/main.go` — the heuristic bot that becomes the GPT-2 policy host
- `scripts/prime_directive_dataset.py` — dataset builder (extended for game replay in M8)
- `scripts/serve.py` — inference server (:8088) — the policy inference endpoint
