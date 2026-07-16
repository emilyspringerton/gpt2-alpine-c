package towerprint

import (
	"errors"
	"strconv"
	"strings"
	"time"
)

// Fingerprint is the Apple-facing composite: everything the original
// end-to-end pipeline derived from one piece of text, in one
// deterministic, JSON-ready value. Intended use (EMILY BACKLOG S147):
// GPT-2-generated text is fingerprinted and the result attached to an
// Apple as a human-glanceable provenance signature.
type Fingerprint struct {
	// Squished is the deduplicated uppercase letter/digit stream.
	Squished string `json:"squished"`
	// Tower is the width-3, V-normalized, X-padded text tower.
	Tower []string `json:"tower"`
	// Magic is Tower with each letter replaced by its dual VVV grid
	// coordinates.
	Magic []string `json:"magic_tower"`
	// Codze is the dual AZ/ZA gematria signature of the letters.
	Codze Codze `json:"codze"`
}

// Compute fingerprints text through the full original pipeline:
// squish, strip non-letters, then (a) Codze the 26-letter stream and
// (b) V-normalize and build the width-3 tower plus its magic form —
// the same staging pemdas.py used (U2V applied before the tower, the
// plain alphabet kept for gematria). Returns an error if text contains
// no letters.
func Compute(text string) (Fingerprint, error) {
	letters := lettersOnly(Squish(text))
	if letters == "" {
		return Fingerprint{}, errors.New("towerprint: no letters to fingerprint")
	}
	codze, err := Codzeify(letters)
	if err != nil {
		return Fingerprint{}, err
	}
	tower := Tower(U2V(letters), TowerWidth)
	magic, err := MagicTower(tower)
	if err != nil {
		return Fingerprint{}, err
	}
	return Fingerprint{
		Squished: letters,
		Tower:    tower,
		Magic:    magic,
		Codze:    codze,
	}, nil
}

// String renders the fingerprint the way the 2020 pipeline printed it:
// squished stream, blank line, the tower.
func (f Fingerprint) String() string {
	return f.Squished + "\n\n" + TowerString(f.Tower)
}

func lettersOnly(s string) string {
	var b strings.Builder
	for _, r := range s {
		if r >= 'A' && r <= 'Z' {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// FortMinute is the original decimal-time seed: the UTC day divided
// into 864-second units (1/100 of a day), minus one — so it runs -1..98
// over a day, quirk preserved from the 2020 code. The original used it
// to seed GPT-2 sampling, anchoring generation to the moment in time;
// S147-02 may reuse it so a fingerprint's generation seed is itself
// time-meaningful.
func FortMinute(t time.Time) int {
	t = t.UTC()
	secs := t.Hour()*3600 + t.Minute()*60 + t.Second()
	return secs/864 - 1
}

// Seed is the original _seed(): the UTC date (YYYYMMDD) concatenated
// with the FortMinute, e.g. "20260716" + "42" -> "2026071642".
func Seed(t time.Time) string {
	return t.UTC().Format("20060102") + strconv.Itoa(FortMinute(t))
}
