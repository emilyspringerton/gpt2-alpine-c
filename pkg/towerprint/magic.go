package towerprint

import (
	"fmt"
	"strings"
)

// SeqVVV is the 24-letter V-normalized alphabet (U and W collapsed into
// V) used by the magic tower. 24 divides evenly into a 3x8 grid, which
// is what makes the dual coordinates below perfectly mirror-symmetric —
// the property the 26-letter alphabet cannot have.
const SeqVVV = "ABCDEFGHIJKLMNOPQRSTVXYZ"

// magicCode maps each SeqVVV letter to its dual grid coordinate pair.
//
// Derivation (recovered from the original magicVVVLookup table, which
// was hand-written in the 2020 notebook): lay SeqVVV into a 3-row x
// 8-column grid column-major (A/B/C down column 0, D/E/F column 1, ...
// — this is gpx(VVV, 3) from the original). A letter at (row r, col c),
// rows counted from the top, gets:
//
//	first pair:  row counted from the BOTTOM, col from the left  = (2-r, c)
//	second pair: row counted from the top, col from the RIGHT    = (r, 7-c)
//
// i.e. its address read from the grid's two opposite corners. The two
// pairs are exact complements ((2-r, 7-c) mirror), making this the 2D
// generalization of Codze's AZ/ZA dual encoding. The generated table is
// verified against the original hand-written one in the tests.
var magicCode = buildMagicCodes()

func buildMagicCodes() map[rune]string {
	m := make(map[rune]string, len(SeqVVV))
	for j, r := range SeqVVV {
		row, col := j%3, j/3
		m[r] = fmt.Sprintf("%d%d %d%d", 2-row, col, row, 7-col)
	}
	return m
}

// MagicCode returns the dual grid coordinate string for a letter (e.g.
// 'A' -> "20 07"). Input is uppercased and V-normalized (U and W map to
// V's code). ok is false for anything outside A-Z.
func MagicCode(r rune) (code string, ok bool) {
	if r >= 'a' && r <= 'z' {
		r -= 'a' - 'A'
	}
	if r == 'U' || r == 'W' {
		r = 'V'
	}
	code, ok = magicCode[r]
	return code, ok
}

// MagicTower converts tower rows (as produced by Tower) into the
// original magicVVVDecTower() form: each letter replaced by its dual
// coordinate code followed by a space, so a width-3 row like "DIN"
// becomes "21 06 02 25 14 13 ". Rows must contain only letters (the 'X'
// padding is a letter and encodes like any other).
func MagicTower(rows []string) ([]string, error) {
	out := make([]string, 0, len(rows))
	for _, row := range rows {
		var b strings.Builder
		for _, r := range row {
			code, ok := MagicCode(r)
			if !ok {
				return nil, fmt.Errorf("towerprint: magic tower: %q has no VVV grid coordinate", r)
			}
			b.WriteString(code)
			b.WriteByte(' ')
		}
		out = append(out, b.String())
	}
	return out, nil
}
