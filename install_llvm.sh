#! /bin/bash

export LLVM_BUILD_DIR=/var/lib/jenkins/llvm-project/build

LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib LLVM_SYSPATH=$LLVM_BUILD_DIR pip install  -e .
