#!/bin/bash -xve

# Dictionary decoding needs only the zstd decompressor.

ZSTD=/usr/src/zstd/lib

compile_options=(
    -O3
    -DNDEBUG
    -DZSTD_DISABLE_ASM
    -DZSTD_LEGACY_SUPPORT=0
    -I$ZSTD -I$ZSTD/common
    $ZSTD/common/*.c $ZSTD/decompress/*.c
    --no-entry
    -s FILESYSTEM=0
    -s ALLOW_MEMORY_GROWTH=1
    -s EXPORTED_FUNCTIONS='["_ZSTD_createDCtx","_ZSTD_freeDCtx","_ZSTD_createDDict","_ZSTD_freeDDict","_ZSTD_decompress_usingDDict","_ZSTD_isError","_malloc","_free"]'
    -s MALLOC=emmalloc
    -s ENVIRONMENT=worker
    -s STANDALONE_WASM=1
    -o zstd_dictionary.wasm
)
emcc ${compile_options[@]}
