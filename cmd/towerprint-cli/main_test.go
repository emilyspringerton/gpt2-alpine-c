package main

import (
	"bytes"
	"encoding/json"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

// buildCLI compiles the binary once per test run into a temp dir so the
// test exercises the real subprocess boundary (stdin/stdout/exit code)
// exactly the way scripts/prime_directive_dataset.py's subprocess.run call
// does, not just the in-process Go functions.
func buildCLI(t *testing.T) string {
	t.Helper()
	_, thisFile, _, _ := runtime.Caller(0)
	pkgDir := filepath.Dir(thisFile)
	binPath := filepath.Join(t.TempDir(), "towerprint-cli")
	cmd := exec.Command("go", "build", "-o", binPath, pkgDir)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build towerprint-cli: %v\n%s", err, out)
	}
	return binPath
}

func TestMain_ValidInput_WritesFingerprintJSON(t *testing.T) {
	bin := buildCLI(t)
	cmd := exec.Command(bin)
	cmd.Stdin = bytes.NewBufferString("hello world")
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	var fp struct {
		Squished string   `json:"squished"`
		Tower    []string `json:"tower"`
		Magic    []string `json:"magic_tower"`
	}
	if err := json.Unmarshal(out, &fp); err != nil {
		t.Fatalf("unmarshal output %q: %v", out, err)
	}
	if fp.Squished == "" {
		t.Error("expected non-empty Squished field")
	}
	if len(fp.Tower) == 0 {
		t.Error("expected non-empty Tower field")
	}
}

func TestMain_EmptyInput_ExitsNonZeroWithErrorJSON(t *testing.T) {
	bin := buildCLI(t)
	cmd := exec.Command(bin)
	cmd.Stdin = bytes.NewBufferString("")
	out, err := cmd.Output()

	if err == nil {
		t.Fatal("expected non-zero exit for input with no letters, got success")
	}
	var errObj struct {
		Error string `json:"error"`
	}
	if jsonErr := json.Unmarshal(out, &errObj); jsonErr != nil {
		t.Fatalf("expected JSON error object on stdout, got %q (unmarshal: %v)", out, jsonErr)
	}
	if errObj.Error == "" {
		t.Error("expected a non-empty error message")
	}
}

func TestMain_NumericOnlyInput_ExitsNonZero(t *testing.T) {
	bin := buildCLI(t)
	cmd := exec.Command(bin)
	cmd.Stdin = bytes.NewBufferString("12345 67890")
	_, err := cmd.Output()
	if err == nil {
		t.Fatal("expected non-zero exit for input with no letters (digits only), got success")
	}
}

func TestMain_Deterministic_SameInputSameOutput(t *testing.T) {
	bin := buildCLI(t)

	run := func() string {
		cmd := exec.Command(bin)
		cmd.Stdin = bytes.NewBufferString("Emily Prime deterministic check")
		out, err := cmd.Output()
		if err != nil {
			t.Fatalf("run: %v", err)
		}
		return string(out)
	}

	first := run()
	second := run()
	if first != second {
		t.Errorf("same input produced different output across runs:\n%s\nvs\n%s", first, second)
	}
}
