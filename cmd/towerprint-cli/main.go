// Command towerprint-cli is the Python↔Go interop point for S150-01:
// prime_directive_dataset.py shells out to this binary rather than
// reimplementing pkg/towerprint's squish/tower/gematria logic in Python,
// which would duplicate and eventually drift from the Go source of truth
// (the same reasoning internal/fabledata's own doc comment gives for
// keeping a house-pattern boundary rather than re-deriving shared logic).
//
// Reads text from stdin, computes towerprint.Compute(text), writes the
// Fingerprint as one line of JSON to stdout. Exit 0 on success; exit 1
// with a JSON {"error": "..."} line on failure (e.g. no letters in the
// input) -- callers should treat a non-zero exit as "skip this record,"
// not a fatal error, mirroring how the rest of the dataset builder
// tolerates individual bad records.
package main

import (
	"encoding/json"
	"io"
	"os"

	"gpt2-alpine-c/pkg/towerprint"
)

func main() {
	data, err := io.ReadAll(os.Stdin)
	if err != nil {
		json.NewEncoder(os.Stdout).Encode(map[string]string{"error": err.Error()})
		os.Exit(1)
	}

	fp, err := towerprint.Compute(string(data))
	if err != nil {
		json.NewEncoder(os.Stdout).Encode(map[string]string{"error": err.Error()})
		os.Exit(1)
	}

	if err := json.NewEncoder(os.Stdout).Encode(fp); err != nil {
		os.Exit(1)
	}
}
