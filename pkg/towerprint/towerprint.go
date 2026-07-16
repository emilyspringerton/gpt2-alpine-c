// Package towerprint is the Go port of the squish/tower/gematria text
// fingerprint recovered from the founder's 2020 experimental repo
// (QUEENSALLYONLINEBOOKOFMAGIFICATIONANDUNICOR — "the magic tower
// transformer"). It turns arbitrary text into a deterministic, human-
// glanceable signature: a squished letter stream, a fixed-width text
// "tower", a dual (AZ/ZA mirror) gematria integer in decimal and binary,
// and a dual-coordinate "magic" tower over a 24-letter V-normalized
// alphabet.
//
// The transform is a gut-check fingerprint, not a security primitive:
// two Apples whose generated content differs will almost always show
// visibly different towers and codze values at a glance, but nothing
// here is collision-resistant in the cryptographic sense.
//
// Behavior is bit-for-bit compatible with the reference Python
// (gpt-2/pemdas.py and TOYBOK/COR.ipynb in the archived repo); the tests
// pin vectors generated from that original code, including the worked
// example saved in the repo's VOIDONX artifact. Design doc:
// docs/TOWERPRINT.md.
package towerprint

import "strings"

// TowerWidth is the canonical tower width used throughout the original
// pipeline (the Python source carries a "TODO CONFIG TOWER WIDTH" but
// every real invocation uses 3, matching the 3-row magic VVV grid).
const TowerWidth = 3

// Squish uppercases s, keeps only [A-Z0-9_], and collapses consecutive
// duplicate kept characters. Dropped characters do not reset the
// duplicate window, exactly matching the original squished(): "a a"
// squishes to "A", "BANANNA" to "BANANA".
func Squish(s string) string {
	var b strings.Builder
	prev := rune(-1)
	for _, r := range strings.ToUpper(s) {
		if !isKept(r) {
			continue
		}
		if r == prev {
			continue
		}
		b.WriteRune(r)
		prev = r
	}
	return b.String()
}

func isKept(r rune) bool {
	return (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_'
}

// U2V uppercases s and collapses U and W into V, producing the 24-letter
// "VVV" alphabet view used by the magic tower (classical-Latin style:
// V = U = W). All other characters pass through unchanged.
func U2V(s string) string {
	var b strings.Builder
	for _, r := range strings.ToUpper(s) {
		if r == 'U' || r == 'W' {
			b.WriteRune('V')
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

// Tower squishes s and chunks it into rows of the given width, padding a
// short final row with 'X'. This is the evolved trxtwr() from the
// notebook (parametric width, X-fill padding), the version the original
// end-to-end pipeline actually fed back into GPT-2. A width below 1 is
// treated as 1.
func Tower(s string, width int) []string {
	if width < 1 {
		width = 1
	}
	sq := Squish(s)
	if sq == "" {
		return nil
	}
	rows := make([]string, 0, (len(sq)+width-1)/width)
	for start := 0; start < len(sq); start += width {
		end := start + width
		if end > len(sq) {
			end = len(sq)
		}
		row := sq[start:end]
		if len(row) < width {
			row += strings.Repeat("X", width-len(row))
		}
		rows = append(rows, row)
	}
	return rows
}

// MatrixTower squishes s and returns one character per row — the
// original MTRXTWER() ("matrix tower"), equivalent to Tower(s, 1).
func MatrixTower(s string) []string {
	return Tower(s, 1)
}

// ClassicTower is the original PRINTWR() row layout: fixed width 3, but
// with the older padding scheme — a lone final character is padded with
// "XZ", a two-character final row with "X". Kept for exact compatibility
// with early saved output; new code should prefer Tower(s, TowerWidth).
func ClassicTower(s string) []string {
	sq := Squish(s)
	if sq == "" {
		return nil
	}
	var rows []string
	for start := 0; start < len(sq); start += TowerWidth {
		end := start + TowerWidth
		if end > len(sq) {
			end = len(sq)
		}
		row := sq[start:end]
		switch len(row) {
		case 1:
			row += "XZ"
		case 2:
			row += "X"
		}
		rows = append(rows, row)
	}
	return rows
}

// TowerString joins tower rows with newlines, with a trailing newline,
// matching the original trxtwrstr() — this is the exact string form the
// 2020 pipeline fed back into GPT-2 as a prompt.
func TowerString(rows []string) string {
	if len(rows) == 0 {
		return ""
	}
	return strings.Join(rows, "\n") + "\n"
}
