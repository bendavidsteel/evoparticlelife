#!/usr/bin/env bash
set -euo pipefail

# Build WASM with Trunk (wasm-opt disabled — Trunk's invocation lacks required feature flags)
trunk build --release

# Run wasm-opt manually with modern WASM feature flags
WASM_FILE=$(ls dist/*_bg.wasm)
echo "Running wasm-opt on $WASM_FILE..."
wasm-opt \
    --enable-bulk-memory \
    --enable-nontrapping-float-to-int \
    --enable-sign-ext \
    --enable-mutable-globals \
    -Oz \
    -o "${WASM_FILE}.opt" \
    "$WASM_FILE"
mv "${WASM_FILE}.opt" "$WASM_FILE"
echo "Done: $(du -h "$WASM_FILE" | cut -f1)"
