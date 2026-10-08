#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "tritonamdgpu-warp-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUPRESERVEWARPPIPELINEOWNERS
#define GEN_PASS_DEF_TRITONAMDGPUWARPPIPELINE
#include "TritonAMDGPUTransforms/Passes.h.inc"

static constexpr llvm::StringLiteral kOwnerBegin =
    "triton.warp_pipeline.owner_begin";
static constexpr llvm::StringLiteral kOwnerEnd =
    "triton.warp_pipeline.owner_end";
static constexpr llvm::StringLiteral kFlatPipeline =
    "triton.warp_pipeline.flat";
static constexpr llvm::StringLiteral kOwner = "triton.warp_pipeline.owner";

// Ops that may appear between pipeline stages but never inside one.  Pre-
// existing memory-fence/wait ops at cluster boundaries are tolerated so that
// prefetch patterns continue to work; encountering one mid-cluster is treated
// as malformed input by the callers.
static bool canSitBetweenStages(Operation *op) {
  return isa<ttg::AsyncWaitOp, gpu::BarrierOp, triton::gpu::BarrierOp,
             tt::amdgpu::AsyncTDMWait>(op);
}

// True if `op` carries the cluster-end marker emitted by the frontend.
static bool isPipelineBorder(Operation *op) {
  return op->hasAttr("triton.warp_pipeline.border");
}

// True if `op` is a structured loop (scf.for / scf.while).  Pipeline clusters
// are straight-line scheduling units, so loops remain boundaries instead of
// being absorbed.  This also rejects nested warp-pipelined scf.for ops.
static bool isLoopOp(Operation *op) {
  return isa<scf::ForOp, scf::WhileOp>(op);
}

// Outcome of attempting to build a pipeline from a region.
//   NotApplicable: no border markers were present (the region opted out).
//   Created:       a pipeline was successfully materialized.
//   Malformed:     border markers were present but the pipeline could not be
//                  built; an error has been emitted at the offending op.
enum class PipelineResult { NotApplicable, Created, Malformed };

// Read (cluster-name, priority) from a border marker op.  Priority defaults
// to -1 when the marker doesn't carry the optional priority attribute.
static std::pair<StringAttr, int> readBorderMarker(Operation *op) {
  StringAttr clusterStr =
      op->getAttrOfType<StringAttr>("triton.warp_pipeline.border");
  int priority = -1;
  if (auto intAttr =
          op->getAttrOfType<IntegerAttr>("triton.warp_pipeline.priority"))
    priority = intAttr.getInt();
  return {clusterStr, priority};
}

// If `cluster` is empty, materialize a dummy SchedBarrier so the cluster is
// non-empty.  This lets users deliberately request a pipeline bubble by
// emitting two consecutive border markers with no body between them.
static void addDummyOpIfEmptyCluster(OpBuilder &b, Location loc,
                                     Operation *insertBefore,
                                     SmallVectorImpl<Operation *> &cluster) {
  if (!cluster.empty())
    return;
  b.setInsertionPoint(insertBefore);
  auto dummyOp =
      ROCDL::SchedBarrier::create(b, loc, ROCDL::SchedGroupMask::none);
  dummyOp->setAttr("triton.warp_pipeline.empty_cluster", b.getUnitAttr());
  cluster.push_back(dummyOp);
}

// Create a scf.execute_region op representing a pipeline cluster.
static void createClusterOp(OpBuilder &b, Location loc,
                            SmallVector<Operation *> &ops,
                            std::pair<StringAttr, int> marker,
                            Attribute flatPipeline = {}) {
  assert(!ops.empty() && "empty stage");

  // Insert the execute_region before the first op in the cluster.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(ops.front());

  // Build fast ops lookup for the cluster.
  llvm::SmallPtrSet<Operation *, 32> opsLookup(ops.begin(), ops.end());

  // Determine which results have users outside the cluster.
  SmallVector<Type> yieldedTypes;
  SmallVector<std::pair<OpResult, unsigned>>
      resultToYieldIdx; // (orig result, idx in yields)

  for (Operation *op : ops) {
    for (OpResult r : op->getResults()) {
      bool hasExternalUse = llvm::any_of(r.getUsers(), [&](Operation *u) {
        return !opsLookup.contains(u) && u->getParentOp() != nullptr;
      });
      if (hasExternalUse) {
        yieldedTypes.push_back(r.getType());
        resultToYieldIdx.emplace_back(
            r, static_cast<unsigned>(yieldedTypes.size() - 1));
      }
    }
  }

  // Create the execute_region with the final result types.
  auto exec = scf::ExecuteRegionOp::create(b, loc, yieldedTypes);
  Block *body = &exec.getRegion().emplaceBlock();
  b.setInsertionPointToStart(body);

  // Clone ops in order, remapping intra-cluster defs to their clones.
  IRMapping mapping;
  for (Operation *op : ops) {
    Operation *clone = b.clone(*op, mapping);
    // Map each result so subsequent clones use the cloned defs.
    for (auto [origRes, clonedRes] :
         llvm::zip(op->getResults(), clone->getResults()))
      mapping.map(origRes, clonedRes);
  }

  // Build the yield values.
  SmallVector<Value> yieldVals(yieldedTypes.size());
  for (auto [origRes, yieldIdx] : resultToYieldIdx) {
    Value mapped = mapping.lookupOrNull(origRes);
    assert(mapped && "mapped result missing");
    yieldVals[yieldIdx] = mapped;
  }
  scf::YieldOp::create(b, loc, yieldVals);

  // Replace external uses of original results with exec results.
  // Internal uses were already remapped when cloning.
  for (auto [origRes, yieldIdx] : resultToYieldIdx) {
    Value repl = exec.getResult(yieldIdx);
    origRes.replaceUsesWithIf(repl, [&](OpOperand &use) {
      Operation *owner = use.getOwner();
      return owner && owner->getParentOp() != exec;
    });
  }

  // Erase original ops now that their external uses are redirected.
  std::reverse(ops.begin(), ops.end());
  for (Operation *op : ops)
    op->erase();

  // Keep the region structured for later conversion.
  exec.setNoInline(true);
  exec->setAttr("triton.warp_pipeline.stage", marker.first);
  if (flatPipeline)
    exec->setAttr(kFlatPipeline, flatPipeline);
  if (marker.second > -1) {
    exec->setAttr("triton.warp_pipeline.priority",
                  b.getI32IntegerAttr(marker.second));
  }

  LLVM_DEBUG(llvm::dbgs() << "[warp-pipeline] created stage with " << ops.size()
                          << " ops and " << yieldedTypes.size() << " yields\n");
  return;
}

// Move pure scalar IV-remap ops after adjacent inter-stage barriers/waits so
// they become part of the next stage.  If a barrier/wait uses one of those
// scalars, leave the run in place to preserve SSA.
static void sinkPureScalarsIntoNextStage(Operation *begin,
                                         Operation *end = nullptr) {
  SmallVector<Operation *> pending;
  auto consumesPending = [&](Operation *user) {
    return llvm::any_of(user->getOperands(), [&](Value v) {
      return llvm::is_contained(pending, v.getDefiningOp());
    });
  };
  for (Operation *op = begin; op && op != end;) {
    Operation *next = op->getNextNode();
    if (triton::isPureScalarOp(op)) {
      pending.push_back(op);
      op = next;
      continue;
    }
    if (canSitBetweenStages(op) && !pending.empty()) {
      Operation *anchor = op;
      while (anchor->getNextNode() && anchor->getNextNode() != end &&
             canSitBetweenStages(anchor->getNextNode()))
        anchor = anchor->getNextNode();
      bool conflict = false;
      for (Operation *ign = op; !conflict; ign = ign->getNextNode()) {
        conflict = consumesPending(ign);
        if (ign == anchor)
          break;
      }
      if (!conflict) {
        next = anchor->getNextNode();
        // Reverse iteration + moveAfter(anchor) preserves source order:
        // each earlier-inserted scalar is pushed right by later inserts.
        for (Operation *s : llvm::reverse(pending))
          s->moveAfter(anchor);
      }
    }
    pending.clear();
    op = next;
  }
}

// Turns a partitioned region into the warp-pipelined clusters.  Returns
// NotApplicable when the loop has no border markers (user opted out), Created
// on success, or Malformed when border markers are present but the loop body
// cannot be split into a valid pipeline (an error is emitted in that case).
static PipelineResult createPipeline(OpBuilder &b, Location loc,
                                     scf::ForOp forOp) {
  Block &blk = *forOp.getBody();

  // Opt-in gate: if the loop body has no borders, the user did not request
  // warp-pipelining for this loop and we must leave it untouched.
  if (llvm::none_of(blk, [](Operation &op) { return isPipelineBorder(&op); }))
    return PipelineResult::NotApplicable;

  SmallVector<Operation *> cluster;
  SmallVector<std::pair<StringAttr, int>> clusterMarkers;
  SmallVector<SmallVector<Operation *>> clusters;
  auto ctx = forOp.getContext();

  sinkPureScalarsIntoNextStage(&blk.front());

  // One pass over the body; collect clusters split by explicit borders.
  for (Operation &opRef : llvm::make_early_inc_range(blk)) {
    Operation *op = &opRef;
    if (isPipelineBorder(op)) { // Wrap up one cluster at a border.
      clusterMarkers.push_back(readBorderMarker(op));
      addDummyOpIfEmptyCluster(b, loc, op, cluster);
      clusters.push_back(std::move(cluster));
      cluster.clear();
      op->erase(); // Remove the marker.
      continue;
    }
    if (canSitBetweenStages(op)) {
      // Barrier / async_wait family ops belong between stages,
      // never inside one.  Encountering one while a cluster is being built
      // means the user inserted it inside a warp_pipeline_stage region.
      if (!cluster.empty()) {
        op->emitError("barrier or wait op cannot appear inside a "
                      "warp_pipeline_stage region");
        return PipelineResult::Malformed;
      }
      continue;
    }
    if (isLoopOp(op)) {
      // Loops are not permitted inside a stage; see isLoopOp for rationale.
      op->emitError("loop op cannot appear inside a warp_pipeline_stage "
                    "region; to pipeline loop iterations, place "
                    "warp_pipeline_stage blocks inside the loop body");
      return PipelineResult::Malformed;
    }
    if (isa<scf::YieldOp>(op)) // End of the loop.
      break;

    // Keep collecting ops for the current cluster.
    cluster.push_back(op);
  }
  if (!cluster.empty()) { // Create the last cluster if needed.
    clusters.push_back(std::move(cluster));
    auto clusterStr = StringAttr::get(ctx, "last_cluster");
    clusterMarkers.push_back({clusterStr, -1});
  }

  // We only reach here when at least one border existed; a single cluster
  // means the borders are degenerate (e.g. a lone trailing border with no
  // operations after it).  Treat as malformed user input.
  if (clusters.size() < 2) {
    forOp->emitError(
        "warp_pipeline_stage borders did not produce at least two stages");
    return PipelineResult::Malformed;
  }

  // Materialize each cluster as an execute_region.
  int totalStages = clusters.size();
  for (auto &&[stageOps, marker] : llvm::zip(clusters, clusterMarkers)) {
    if (stageOps.empty())
      continue;
    createClusterOp(b, loc, stageOps, marker);
  }

  // Annotate the loop for the backend.
  b.setInsertionPoint(forOp);
  forOp->setAttr("triton.warp_pipeline.pipelined_for", b.getUnitAttr());

  LDBG("[warp-pipeline] total_stages=" << totalStages << "\n");
  return PipelineResult::Created;
}

// Create a pipelined region from flat (non-loop) border markers in a block.
// This handles the case where a loop was unrolled at the Python level
// (e.g. via static_range) but the body still has warp_pipeline_stage
// annotations producing border markers.  The grouping logic mirrors
// createPipeline but without a loop wrapper.
//
// Returns NotApplicable when the block has no border markers, Created when
// a flat pipeline was materialized, or Malformed when borders are present
// but a valid pipeline could not be built (an error is emitted in that case).
static PipelineResult createFlatPipeline(OpBuilder &b, Block &block,
                                         Attribute flatPipeline,
                                         Operation *scopeBegin = nullptr,
                                         Operation *scopeEnd = nullptr) {
  // 1. Find all border markers in this block.
  SmallVector<Operation *> allBorders;
  Operation *first = scopeBegin ? scopeBegin->getNextNode() : &block.front();
  for (Operation *op = first; op && op != scopeEnd; op = op->getNextNode())
    if (isPipelineBorder(op))
      allBorders.push_back(op);

  // No borders at all means the block did not opt into flat pipelining.
  if (allBorders.empty())
    return PipelineResult::NotApplicable;

  // Without an owner boundary, a single border cannot form two flat stages.
  // A bounded scope can include a second stage after its last border.
  if (!scopeEnd && allBorders.size() < 2) {
    allBorders.front()->emitError(
        "warp_pipeline_stage requires at least two borders to form a flat "
        "pipeline");
    return PipelineResult::Malformed;
  }

  Location loc = allBorders.front()->getLoc();
  Operation *firstBorder = allBorders.front();
  Operation *lastBorder = allBorders.back();

  // 2. For flat pipelines, stage 0 may include the ops immediately before the
  //    first border.  Stop at ops that must stay outside this pipeline.
  Operation *regionStart = first;
  if (!scopeBegin) {
    regionStart = firstBorder;
    for (Operation *op = firstBorder->getPrevNode(); op;
         op = op->getPrevNode()) {
      if (isLoopOp(op) || isa<tt::amdgpu::CondBarrierOp>(op) ||
          canSitBetweenStages(op))
        break;
      regionStart = op;
    }
  }

  sinkPureScalarsIntoNextStage(regionStart, scopeEnd);
  if (scopeBegin)
    regionStart = scopeBegin->getNextNode();

  // 3. Sweep forward from regionStart, splitting ops into clusters at each
  //    border. Stop at the owner boundary, or the last border for legacy input.
  SmallVector<Operation *> cluster;
  SmallVector<std::pair<StringAttr, int>> clusterMarkers;
  SmallVector<SmallVector<Operation *>> clusters;

  for (auto it = Block::iterator(regionStart); it != block.end();) {
    Operation *op = &*it;
    ++it;

    if (op == scopeEnd)
      break;

    if (isPipelineBorder(op)) {
      clusterMarkers.push_back(readBorderMarker(op));
      addDummyOpIfEmptyCluster(b, loc, op, cluster);
      clusters.push_back(std::move(cluster));
      cluster.clear();

      bool isLast = !scopeEnd && (op == lastBorder);
      op->erase();
      if (isLast)
        break;
      continue;
    }

    if (canSitBetweenStages(op)) {
      // Same rule as createPipeline: barriers/waits cannot live inside a
      // stage.
      if (!cluster.empty()) {
        op->emitError("barrier or wait op cannot appear inside a "
                      "warp_pipeline_stage region");
        return PipelineResult::Malformed;
      }
      continue;
    }

    if (isLoopOp(op)) {
      // Same rule as createPipeline: loops cannot live inside a stage.
      op->emitError("loop op cannot appear inside a warp_pipeline_stage "
                    "region; to pipeline loop iterations, place "
                    "warp_pipeline_stage blocks inside the loop body");
      return PipelineResult::Malformed;
    }

    cluster.push_back(op);
  }

  if (scopeEnd && !cluster.empty()) {
    clusters.push_back(std::move(cluster));
    clusterMarkers.emplace_back(b.getStringAttr("last_cluster"), -1);
  }

  // 4. The bounded sweep should produce at least two clusters.
  if (clusters.size() < 2) {
    mlir::emitError(
        loc, "warp_pipeline_stage borders did not produce at least two stages");
    return PipelineResult::Malformed;
  }

  for (auto &&[stageOps, marker] : llvm::zip(clusters, clusterMarkers)) {
    if (stageOps.empty())
      continue;
    createClusterOp(b, loc, stageOps, marker, flatPipeline);
  }

  LDBG("[warp-pipeline] flat pipeline with " << clusters.size() << " stages");
  return PipelineResult::Created;
}

struct TritonAMDGPUPreserveWarpPipelineOwnersPass
    : impl::TritonAMDGPUPreserveWarpPipelineOwnersBase<
          TritonAMDGPUPreserveWarpPipelineOwnersPass> {
  void runOnOperation() override {
    OpBuilder builder(getOperation());
    getOperation().walk([&](scf::ForOp loop) {
      if (loop->hasAttr(kOwner) ||
          llvm::none_of(*loop.getBody(),
                        [](Operation &op) { return isPipelineBorder(&op); }))
        return;
      loop->setAttr(kOwner, builder.getUnitAttr());
      // Parent-block markers survive when the loop body is spliced or cloned.
      // Reuse scheduling barriers so canonicalization retains the boundaries.
      builder.setInsertionPoint(loop);
      auto begin = ROCDL::SchedBarrier::create(builder, loop.getLoc(),
                                               ROCDL::SchedGroupMask::none);
      begin->setAttr(kOwnerBegin, builder.getUnitAttr());
      builder.setInsertionPointAfter(loop);
      auto end = ROCDL::SchedBarrier::create(builder, loop.getLoc(),
                                             ROCDL::SchedGroupMask::none);
      end->setAttr(kOwnerEnd, builder.getUnitAttr());
    });
  }
};

// Recover folded owners before looking for borders in surviving loop bodies.
// A flat pipeline keeps its own identity, including beside another flat owner.
static LogicalResult recoverOwnerScopes(OpBuilder &builder, tt::FuncOp func,
                                        int64_t &nextFlatPipeline) {
  SmallVector<Block *> blocks;
  func.walk<WalkOrder::PostOrder>(
      [&](Block *block) { blocks.push_back(block); });
  for (Block *block : blocks) {
    SmallVector<std::pair<Operation *, Operation *>> scopes;
    Operation *begin = nullptr;
    for (Operation &op : *block) {
      if (op.hasAttr(kOwnerBegin)) {
        if (begin)
          return op.emitError("nested warp-pipeline owner scopes are not "
                              "supported");
        begin = &op;
      } else if (op.hasAttr(kOwnerEnd)) {
        if (!begin)
          return op.emitError("unmatched warp-pipeline owner end marker");
        scopes.emplace_back(begin, &op);
        begin = nullptr;
      }
    }
    if (begin)
      return begin->emitError("unmatched warp-pipeline owner begin marker");

    for (auto [scopeBegin, scopeEnd] : scopes) {
      // Partial unrolling can leave a surviving owner followed by a flat tail.
      // Partition at those loops, but still reject loops inside a flat stage.
      SmallVector<Operation *> boundaries;
      for (Operation *op = scopeBegin->getNextNode(); op != scopeEnd;
           op = op->getNextNode())
        if (isa<scf::ForOp>(op) && op->hasAttr(kOwner))
          boundaries.push_back(op);
      boundaries.push_back(scopeEnd);

      Operation *segmentBegin = scopeBegin;
      for (Operation *segmentEnd : boundaries) {
        auto group = builder.getI64IntegerAttr(nextFlatPipeline++);
        if (createFlatPipeline(builder, *block, group, segmentBegin,
                               segmentEnd) == PipelineResult::Malformed)
          return failure();
        segmentBegin = segmentEnd;
      }
      scopeBegin->erase();
      scopeEnd->erase();
    }
  }
  return success();
}

struct TritonAMDGPUWarpPipelinePass
    : impl::TritonAMDGPUWarpPipelineBase<TritonAMDGPUWarpPipelinePass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder builder(m);
    bool malformed = false;
    int64_t nextFlatPipeline = 0;
    for (auto funcOp : m.getOps<tt::FuncOp>()) {
      if (failed(recoverOwnerScopes(builder, funcOp, nextFlatPipeline))) {
        signalPassFailure();
        return;
      }
      funcOp.walk([&](scf::ForOp forOp) {
        forOp->removeAttr(kOwner);
        Location loc = forOp.getLoc();
        switch (createPipeline(builder, loc, forOp)) {
        case PipelineResult::NotApplicable:
          LDBG("scf.for has no warp_pipeline_stage borders; skipping");
          break;
        case PipelineResult::Created:
          break;
        case PipelineResult::Malformed:
          malformed = true;
          break;
        }
      });

      // Recognize legacy unbounded flat pipelines only at function scope.
      // Folded owners above provide explicit bounds for nested pipelines.
      for (Block &block : funcOp.getBody()) {
        switch (createFlatPipeline(
            builder, block, builder.getI64IntegerAttr(nextFlatPipeline++))) {
        case PipelineResult::NotApplicable:
          break;
        case PipelineResult::Created:
          break;
        case PipelineResult::Malformed:
          malformed = true;
          break;
        }
      }
    }
    if (malformed)
      signalPassFailure();
  }
};

} // namespace mlir
