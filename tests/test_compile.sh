\
    #!/usr/bin/env bash
    set -euo pipefail
    make clean
    make
    echo "Running binary with missing weights (expected non-zero exit)..."
    ./gpt2_run weights/model.bin || true
    echo "Compile/run test done."
