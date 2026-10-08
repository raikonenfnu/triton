// RUN: triton-opt %s -split-input-file -gluon-inline="simplify=false" -tritonamdgpu-preserve-warp-pipeline-owners -gluon-slimplify-control-flow -canonicalize -triton-loop-unroll -canonicalize -tritonamdgpu-warp-pipeline | FileCheck %s --check-prefix=PIPE --implicit-check-not=triton.warp_pipeline.owner --implicit-check-not=triton.warp_pipeline.border
// RUN: triton-opt %s -split-input-file -gluon-inline="simplify=false" -tritonamdgpu-preserve-warp-pipeline-owners -gluon-slimplify-control-flow -canonicalize -triton-loop-unroll -canonicalize -tritonamdgpu-warp-pipeline -convert-warp-pipeline="gfx-arch=gfx950" | FileCheck %s --check-prefix=LOWER --implicit-check-not=scf.execute_region --implicit-check-not=triton.warp_pipeline

// Inlining exposes stage borders before simplification folds their owner.
// The returned accumulator also tests SSA uses outside the folded scope.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func private @stages(%ptr: !tt.ptr<i32>, %acc: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    tt.store %ptr, %acc : !tt.ptr<i32>
    rocdl.sched.barrier none {triton.warp_pipeline.border = "first"}
    %next = arith.addi %acc, %c1 : i32
    tt.store %ptr, %next : !tt.ptr<i32>
    rocdl.sched.barrier none {triton.warp_pipeline.border = "second"}
    tt.return %next : i32
  }

  tt.func @single_trip_helper(%n: i32, %ptr: !tt.ptr<i32>, %seed: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %tile = %c0 to %n step %c1 : i32 {
      ttg.async_wait {num = 1 : i32}
      %result = scf.for %k = %c0 to %c1 step %c1 iter_args(%acc = %seed) -> i32 : i32 {
        %next = tt.call @stages(%ptr, %acc) : (!tt.ptr<i32>, i32) -> i32
        scf.yield %next : i32
      }
      ttg.async_wait {num = 0 : i32}
      tt.store %ptr, %result : !tt.ptr<i32>
    }
    tt.return
  }
}

// PIPE-LABEL: tt.func @single_trip_helper
// PIPE: scf.for
// PIPE: ttg.async_wait {num = 1
// PIPE-NEXT: scf.execute_region
// PIPE: triton.warp_pipeline.flat = [[GROUP:[0-9]+]] : i64
// PIPE: [[RESULT:%.*]] = scf.execute_region -> i32
// PIPE: arith.addi
// PIPE: scf.yield
// PIPE: triton.warp_pipeline.flat = [[GROUP]] : i64
// PIPE-NEXT: {{.*}}ttg.async_wait {num = 0
// PIPE-NEXT: tt.store {{.*}}, [[RESULT]]
// PIPE-NOT: triton.warp_pipeline.pipelined_for
// PIPE: tt.return

// LOWER-LABEL: tt.func @single_trip_helper
// LOWER: ttg.async_wait {num = 1
// LOWER: amdg.cond_barrier
// LOWER: tt.store
// LOWER: tt.store
// LOWER: amdg.cond_barrier
// LOWER: ttg.async_wait {num = 0
// LOWER: tt.store
// LOWER: tt.return

// -----

// Without a prologue wait, selecting the wrong loop would silently compile.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @single_trip_without_wait(%n: i32, %ptr: !tt.ptr<i32>, %value: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %tile = %c0 to %n step %c1 : i32 {
      tt.store %ptr, %value : !tt.ptr<i32>
      scf.for %k = %c0 to %c1 step %c1 : i32 {
        tt.store %ptr, %c0 : !tt.ptr<i32>
        rocdl.sched.barrier none {triton.warp_pipeline.border = "first"}
        tt.store %ptr, %c1 : !tt.ptr<i32>
        rocdl.sched.barrier none {triton.warp_pipeline.border = "second"}
      }
      tt.store %ptr, %value : !tt.ptr<i32>
    }
    tt.return
  }
}

// PIPE-LABEL: tt.func @single_trip_without_wait
// PIPE: scf.for
// PIPE-NEXT: tt.store
// PIPE-NEXT: scf.execute_region
// PIPE: triton.warp_pipeline.flat
// PIPE-NEXT: scf.execute_region
// PIPE: triton.warp_pipeline.flat
// PIPE-NEXT: tt.store
// PIPE-NOT: triton.warp_pipeline.pipelined_for
// PIPE: tt.return

// LOWER-LABEL: tt.func @single_trip_without_wait
// LOWER: scf.for
// LOWER-NEXT: tt.store
// LOWER: amdg.cond_barrier
// LOWER: tt.store
// LOWER: tt.store
// LOWER: amdg.cond_barrier
// LOWER-NEXT: tt.store
// LOWER: tt.return

// -----

// Fully unrolled owners retain their scope inside a conditional.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @unrolled_in_if(%cond: i1, %ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    scf.if %cond {
      ttg.async_wait {num = 1 : i32}
      scf.for %k = %c0 to %c2 step %c1 : i32 {
        tt.store %ptr, %k : !tt.ptr<i32>
        rocdl.sched.barrier none {triton.warp_pipeline.border = "first"}
        tt.store %ptr, %c2 : !tt.ptr<i32>
        rocdl.sched.barrier none {triton.warp_pipeline.border = "second"}
      } {tt.loop_unroll_factor = 2 : i32}
      ttg.async_wait {num = 0 : i32}
    }
    tt.return
  }
}

// PIPE-LABEL: tt.func @unrolled_in_if
// PIPE: scf.if
// PIPE: ttg.async_wait {num = 1
// PIPE-COUNT-4: scf.execute_region
// PIPE: ttg.async_wait {num = 0
// PIPE-NOT: scf.for
// PIPE: tt.return

// LOWER-LABEL: tt.func @unrolled_in_if
// LOWER: scf.if
// LOWER: ttg.async_wait {num = 1
// LOWER: amdg.cond_barrier
// LOWER-COUNT-4: tt.store
// LOWER: amdg.cond_barrier
// LOWER: ttg.async_wait {num = 0
// LOWER: tt.return

// -----

// A surviving multi-trip owner still uses the normal loop pipeline.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @multi_trip(%n: i32, %ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %k = %c0 to %n step %c1 : i32 {
      tt.store %ptr, %k : !tt.ptr<i32>
      rocdl.sched.barrier none {triton.warp_pipeline.border = "first"}
      tt.store %ptr, %c0 : !tt.ptr<i32>
      rocdl.sched.barrier none {triton.warp_pipeline.border = "second"}
    }
    tt.return
  }
}

// PIPE-LABEL: tt.func @multi_trip
// PIPE: scf.for
// PIPE-COUNT-2: scf.execute_region
// PIPE-NOT: triton.warp_pipeline.flat
// PIPE: triton.warp_pipeline.pipelined_for
// PIPE: tt.return

// LOWER-LABEL: tt.func @multi_trip
// LOWER: scf.for
// LOWER-COUNT-2: tt.store
// LOWER: tt.return

// -----

// A zero-trip owner removes its borders and bounds, but retains outer work.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @zero_trip(%ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    ttg.async_wait {num = 0 : i32}
    scf.for %k = %c0 to %c0 step %c1 : i32 {
      tt.store %ptr, %k : !tt.ptr<i32>
      rocdl.sched.barrier none {triton.warp_pipeline.border = "first"}
      rocdl.sched.barrier none {triton.warp_pipeline.border = "second"}
    }
    tt.store %ptr, %c1 : !tt.ptr<i32>
    tt.return
  }
}

// PIPE-LABEL: tt.func @zero_trip
// PIPE-NOT: scf.for
// PIPE-NOT: scf.execute_region
// PIPE: ttg.async_wait
// PIPE-NEXT: tt.store
// PIPE-NEXT: tt.return

// LOWER-LABEL: tt.func @zero_trip
// LOWER: ttg.async_wait
// LOWER-NEXT: tt.store
// LOWER-NEXT: tt.return

// -----

// Adjacent folded loops remain distinct owners during flat conversion.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @adjacent_owners(%ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %k = %c0 to %c1 step %c1 : i32 {
      tt.store %ptr, %c0 : !tt.ptr<i32>
      rocdl.sched.barrier none {triton.warp_pipeline.border = "first"}
      tt.store %ptr, %c1 : !tt.ptr<i32>
      rocdl.sched.barrier none {triton.warp_pipeline.border = "second"}
    }
    scf.for %k = %c0 to %c1 step %c1 : i32 {
      tt.store %ptr, %c1 : !tt.ptr<i32>
      rocdl.sched.barrier none {triton.warp_pipeline.border = "first"}
      tt.store %ptr, %c0 : !tt.ptr<i32>
      rocdl.sched.barrier none {triton.warp_pipeline.border = "second"}
    }
    tt.return
  }
}

// PIPE-LABEL: tt.func @adjacent_owners
// PIPE: triton.warp_pipeline.flat = 0 : i64
// PIPE: triton.warp_pipeline.flat = 0 : i64
// PIPE: triton.warp_pipeline.flat = 1 : i64
// PIPE: triton.warp_pipeline.flat = 1 : i64
// PIPE: tt.return

// LOWER-LABEL: tt.func @adjacent_owners
// LOWER: amdg.cond_barrier
// LOWER-COUNT-4: tt.store
// LOWER: amdg.cond_barrier
// LOWER: tt.return

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @unannotated_loop(%n: i32, %ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %k = %c0 to %n step %c1 : i32 {
      tt.store %ptr, %k : !tt.ptr<i32>
    }
    tt.return
  }
}

// PIPE-LABEL: tt.func @unannotated_loop
// PIPE: scf.for
// PIPE-NEXT: tt.store
// PIPE-NOT: triton.warp_pipeline
// PIPE: tt.return

// LOWER-LABEL: tt.func @unannotated_loop
// LOWER: scf.for
// LOWER-NEXT: tt.store
// LOWER: tt.return

// -----

// Partial unrolling leaves a pipelined loop and a bounded single-trip tail.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @partial_unroll(%ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c5 = arith.constant 5 : i32
    ttg.async_wait {num = 1 : i32}
    scf.for %k = %c0 to %c5 step %c1 : i32 {
      tt.store %ptr, %k : !tt.ptr<i32>
      rocdl.sched.barrier none {triton.warp_pipeline.border = "first"}
      tt.store %ptr, %c5 : !tt.ptr<i32>
      rocdl.sched.barrier none {triton.warp_pipeline.border = "second"}
    } {tt.loop_unroll_factor = 2 : i32}
    ttg.async_wait {num = 0 : i32}
    tt.return
  }
}

// PIPE-LABEL: tt.func @partial_unroll
// PIPE: ttg.async_wait {num = 1
// PIPE: scf.for
// PIPE-COUNT-4: scf.execute_region
// PIPE: triton.warp_pipeline.pipelined_for
// PIPE-COUNT-2: triton.warp_pipeline.flat
// PIPE: ttg.async_wait {num = 0
// PIPE: tt.return

// LOWER-LABEL: tt.func @partial_unroll
// LOWER: ttg.async_wait {num = 1
// LOWER: scf.for
// LOWER-COUNT-6: tt.store
// LOWER: ttg.async_wait {num = 0
// LOWER: tt.return
