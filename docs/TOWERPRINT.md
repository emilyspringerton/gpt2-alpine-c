# TOWERPRINT — Squish/Tower/Gematria Fingerprint

**Package:** `gpt2-alpine-c/pkg/towerprint` (Go, stdlib-only, in `go.work`)
**Reference implementation:** `~/QUEENSALLYONLINEBOOKOFMAGIFICATIONANDUNICOR` (archived, 2020)
**Backlog:** EMILY SECTION 147 (Apple enrichment). S147-01 is this package.

---

## 1. What the 2020 repo actually was

Archaeology conclusion, with evidence — not a guess:

The founder's 2020 repo was a **GPT-2 divination loop**: a gematria-style reading
system where the same text is re-encoded at rising levels of abstraction and each
encoding is fed back to GPT-2 as a prompt, using the model as the oracle. The
`interact_model` loop in `gpt-2/pemdas.py` (and the Sleepy-Hollow-driven
`hollow.py`) runs every input through exactly four staged readings:

1. raw text → GPT-2
2. `U2V(squished(text))` (condensed, V-normalized) → GPT-2
3. the width-3 text tower (`trxtwr`/`trxtwrstr`) → GPT-2
4. the magic VVV coordinate tower interleaved with the letter tower → GPT-2

Evidence for the divination/reading framing rather than compression or checksum:

- **The seed is a moment, not entropy.** `fortminute()` divides the UTC day into
  864-second units (1/100 day, decimal time) and seeds sampling with
  `YYYYMMDD + fortminute` — generation is deterministically anchored to *when* the
  reading is cast. That is horoscope mechanics, not RNG hygiene, and it is why
  S147's astrology field belongs in the same feature.
- **AZ/ZA is Atbash.** `codzeifyWord` computes every word's value in the forward
  alphabet *and* the mirrored one — the mirror alphabet is the Atbash cipher, a
  classical Hebrew gematria technique. A checksum needs one direction; a
  numerological reading traditionally wants both.
- **The V-alphabet is epigraphic.** `U2V` collapses U/W into V (classical Latin,
  where V=U=W), leaving 24 letters — chosen because 24 = 3×8 divides evenly,
  which 26 cannot. See §3.
- **The notebook says so.** `TOYBOK/COR.ipynb` cell 17's TODO speaks of "tower
  density", "dualistic density", "nondualistic density" and renders words as
  binary heatmaps — visual signatures of words, not compressed data (every
  transform *loses* information; nothing is decodable).
- **`VOIDONX` is a séance transcript.** The saved artifact shows towers whose
  leading rows exactly match `trxtwr` output (verified in the port's tests:
  `NOW/EAR/EON/LYB/IRD` for "NOW WE ARE ONLY BIRDS") and whose trailing rows
  (`DZXR`, `GZX_STILL`, `LGX`) are GPT-2 *continuing the tower pattern itself* —
  the whole point was to see how the model answers each encoding.

So: **not** a signature/checksum scheme by intent, and not compression — but the
transform happens to have exactly the deterministic, human-glanceable shape a
non-cryptographic fingerprint wants, which is why S147 reuses it.

## 2. Verdict on the two variants

The notebook-only `trxtwr`/`magicVVVDecTower` family **is the evolved, intended
final version** and is the core of the port:

- `trxtwr` parameterizes tower width (resolving `PRINTWR`'s literal
  `TODO CONFIG TOWER WIDTH`) and returns data instead of printing.
- Its X-fill padding replaces `PRINTWR`'s ad-hoc `XZ`/`X` padding.
- It is what `pemdas.py`/`hollow.py`'s real pipeline calls — the `.py` exports
  postdate the notebook and use `trxtwr`, not `PRINTWR`, in stages 3–4.
- The width-3 tower and the 3-row magic grid share one geometry (§3); `MTRXTWER`
  and `PRINTWR` are earlier sketches of the same idea.

Both are ported: `Tower`/`MatrixTower`/`ClassicTower` cover `trxtwr`, `MTRXTWER`,
and `PRINTWR` respectively, with `ClassicTower` kept only for compatibility with
early saved output.

## 3. Transform spec (as ported)

- **`Squish(s)`** — uppercase; keep `[A-Z0-9_]`; collapse consecutive duplicates
  (dropped chars don't reset the window: `"a a"` → `"A"`). Original: `squished()`.
- **`U2V(s)`** — uppercase; U,W → V. Original: `U2V()`.
- **`Tower(s, width)`** — squish, chunk into rows, X-pad the last row.
  Original: `trxtwr()` + `trxtwrstr()` (`TowerString` renders the prompt form).
- **`Codzeify(word)`** — per letter, digit = alphabet index mod 8, read the digit
  string as one arbitrary-precision decimal integer; computed for the forward
  (AZ) and mirrored (ZA) alphabet; decimal + binary renderings. Original:
  `codzeifyWord()` (`math/big` matches Python ints on long input).
- **`MagicCode(r)` / `MagicTower(rows)`** — the 24-letter V-alphabet laid
  column-major into a 3×8 grid; each letter's code is its address read from two
  opposite corners: `(2-row, col)` and `(row, 7-col)` — exact complements, the 2D
  generalization of AZ/ZA. The hand-written `magicVVVLookup` table from the
  notebook is *derived* in the port and test-verified against the original,
  entry for entry.
- **`Compute(text)`** — the composite `Fingerprint`: squished letters, width-3
  V-normalized tower, magic tower, dual codze. Staging matches `pemdas.py`
  (U2V before the tower; plain 26-letter alphabet for gematria).
- **`FortMinute(t)` / `Seed(t)`** — the decimal-time seed, quirks preserved
  (`-1..98` range).

All behavior is pinned by table-driven tests against vectors generated from the
original Python (pure string functions — no TF needed) plus the `VOIDONX`
artifact and executed notebook cell outputs.

## 4. Why this is the right Apple fingerprint shape

An Apple fingerprint here is a **cheap gut-check, not a security primitive**:

- **Deterministic** — same generated text, same fingerprint, forever; no keys,
  no state, no network.
- **Human-glanceable** — two towers side by side visibly match or don't in the
  MJOLNIR feed or `emily apples` output; a sha256 requires exact comparison,
  a tower can be *read*. The codze decimal gives a compact scalar for eyeballing;
  the binary gives the "density" view the notebook was already plotting.
- **Honest about what it is** — collisions are possible and fine; the model
  fingerprint (S147-03) is the provenance field, the tower is the glanceable one.
- **House lineage** — it is the founder's own 2020 construction, revived, which
  is precisely the "emilyify, don't museum-ify" instruction.

## 5. Wiring into S147-02 (the missing middle step)

```
Apple event ──► emily-agent enrichment worker
                  │  POST :8088/generate  {"prompt": <apple title/summary>,
                  │                        "max_tokens": ..., "temperature": ...}
                  │  ◄─ {"text": ..., "model": "emily-ft (checkpoint-...)"}
                  │
                  │  fp, _ := towerprint.Compute(resp.Text)
                  │
                  └─► PATCH IDUNA apple:
                        gpt2_fingerprint: {
                          "generated":  resp.Text,
                          "squished":   fp.Squished,
                          "tower":      fp.Tower,
                          "magic_tower":fp.Magic,
                          "codze":      fp.Codze,          // az/za dec+bin
                          "seed":       towerprint.Seed(t) // decimal-time anchor
                        },
                        model_fingerprint: resp.Model      // S147-03; + weights hash later
```

- The transform runs **in-process in Go** (this package) — serve.py is only
  called for generation. No Python is invoked for the fingerprint itself.
- `serve.py`'s `/generate` already returns the `model` tag, which is the seed of
  S147-03's model fingerprint; a sha256 of the checkpoint weights file can be
  added to serve.py's `/health` later without changing this design.
- Suggested prompt: the Apple's title (short, stable); `towerprint.Seed(now)`
  can be passed through once serve.py accepts a seed parameter, making the
  generation itself time-anchored like the 2020 original.

### Position: **async enrichment** (Apple lands first, fingerprint PATCHed after)

Not sync. Reasons, in priority order:

1. **Apples are the audit trail; enrichment is decoration.** A GPT-2 box being
   down, slow, or mid-restart must never delay or fail an Apple POST. Sync
   coupling inverts the dependency: the trust ledger would depend on an
   experimental inference server.
2. **Latency reality.** CPU GPT-2 generation of ~100 tokens takes seconds;
   Apple POSTs happen on the 5-minute RSI hot path and from the CLI. Blocking
   every filing on that is unacceptable and would pressure people to disable it.
3. **Precedent.** EMILY already runs async workers against IDUNA
   (CheckinAlertWorker); an ApplEnrichWorker is the same shape: poll/queue
   recent Apples missing `gpt2_fingerprint`, generate, PATCH.
4. **Degradation mode is honest.** Async's failure mode is "field missing,
   retried later" — visible and recoverable. Sync's is a lost or delayed Apple.

Consequence for S147-05: enrichment is **caller-side** (emily-agent worker), not
IDUNA-side — IDUNA stays a passive store of optional fields and never calls out
to an inference box. IDUNA's only change is schema: three optional fields +
accepting PATCH of them.

## 6. Out of scope here

- S147-02/03/05 implementation (needs live serve.py + IDUNA handler work).
- S147-04 astrology/transit source — explicitly open; `FortMinute`/`Seed` is the
  only time-anchoring this package provides and is not an ephemeris.
