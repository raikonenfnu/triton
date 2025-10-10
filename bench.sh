#!/bin/bash

# batch_size=(32 16 8 4 2)
# seqlen=(512 1024 2048 4096 8192 16384)

set -euo pipefail

# Read argument
dtype=${1:-}

# Check if provided and valid
if [[ -z "$dtype" ]]; then
    echo "Error: You must provide a data type (fp16 or bf16)." >&2
    exit 1
fi

if [[ "$dtype" != "fp16" && "$dtype" != "bf16" ]]; then
    echo "Error: Invalid data type '$dtype'. Must be one of: fp16 or bf16." >&2
    exit 1
fi

headim=(128)
head_count_q=64
head_count_k=64
causal=(0 1)
layout="bshd" # options: bshd bhsd thd

# BatchSize, SeqLen
configs=(
    "32 512"
    "16 1024"
    "8 2048"
    "4 4096"
    "2 8192"
    "1 16384"
)

# Enable FAv3
export DISABLE_LLVM_OPT="disable-vector-combine" TRITON_HIP_USE_PADDED_SHARED_LAYOUT=1 TRITON_HIP_USE_ASYNC_COPY=1 AMDGCN_SCALARIZE_PACKED_FOPS=1

for c in "${causal[@]}"; do
    for h in "${headim[@]}"; do
        for batchSeq in "${configs[@]}"; do
            read -r b s <<< "$batchSeq"
            #echo "Causal: $c, HeadDim: $h Batch size: $b SeqLen: $s"
            python3 fa/flash-attention.py -d "$h" -hq 64 -b "$b" -sq "$s" -causal "$c" -layout "$layout" -dtype $dtype
        done
    done
done
