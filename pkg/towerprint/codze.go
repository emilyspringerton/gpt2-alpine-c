package towerprint

import (
	"errors"
	"fmt"
	"math/big"
	"strconv"
	"strings"
)

// DefaultGroupLen is the original defaultGrpLen(): the 26-letter alphabet
// dealt round-robin into 8 groups, so each letter's digit is its
// alphabet index mod 8.
const DefaultGroupLen = 8

// Signature is one directional gematria value of a word: the digit
// string read as a decimal integer, rendered in decimal and binary.
// Values grow with input length without bound, so both are strings
// (arbitrary precision, matching Python int).
type Signature struct {
	Dec string `json:"dec"`
	Bin string `json:"bin"`
}

// Codze is the dual gematria signature of a word — the original
// codzeifyWord() output. AZ reads each letter's group index in the
// forward alphabet; ZA reads it in the mirrored (Atbash-style) alphabet.
// A word only matches another in both directions if it is letter-for-
// letter group-identical both forwards and mirrored.
type Codze struct {
	Word string    `json:"word"`
	AZ   Signature `json:"az"`
	ZA   Signature `json:"za"`
}

// Codzeify computes the dual AZ/ZA gematria signature of s with the
// canonical group length of 8. s must be non-empty and contain only
// letters (case-insensitive) — squish and strip digits first.
func Codzeify(s string) (Codze, error) {
	return CodzeifyGrouped(s, DefaultGroupLen)
}

// CodzeifyGrouped is Codzeify with an explicit group length, matching
// the original codzeifyWord(s, grpLen). Each letter contributes the
// decimal string of (index mod grpLen) in the forward alphabet for AZ,
// and of (mirrorIndex mod grpLen) for ZA; the concatenated digit strings
// are read as decimal integers.
func CodzeifyGrouped(s string, grpLen int) (Codze, error) {
	if grpLen < 1 {
		return Codze{}, fmt.Errorf("towerprint: codzeify: group length %d < 1", grpLen)
	}
	up := strings.ToUpper(s)
	if up == "" {
		return Codze{}, errors.New("towerprint: codzeify: empty word")
	}
	var az, za strings.Builder
	for _, r := range up {
		if r < 'A' || r > 'Z' {
			return Codze{}, fmt.Errorf("towerprint: codzeify: %q is not a letter", r)
		}
		j := int(r - 'A')
		az.WriteString(strconv.Itoa(j % grpLen))
		za.WriteString(strconv.Itoa((25 - j) % grpLen))
	}
	return Codze{
		Word: up,
		AZ:   sigFromDigits(az.String()),
		ZA:   sigFromDigits(za.String()),
	}, nil
}

func sigFromDigits(digits string) Signature {
	v, ok := new(big.Int).SetString(digits, 10)
	if !ok {
		// Unreachable: digits is always a non-empty decimal string.
		panic("towerprint: invalid digit string " + strconv.Quote(digits))
	}
	return Signature{Dec: v.String(), Bin: v.Text(2)}
}
