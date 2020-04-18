//===- AArch64StackTagging.cpp - Stack tagging in IR --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "AArch64TargetMachine.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/MD5.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include <cassert>
#include <iterator>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "stack-tagging"

static cl::opt<bool> ClMergeInit(
    "stack-tagging-merge-init", cl::Hidden, cl::init(true), cl::ZeroOrMore,
    cl::desc("merge stack variable initializers with tagging when possible"));

static cl::opt<unsigned> ClScanLimit("stack-tagging-merge-init-scan-limit",
                                     cl::init(40), cl::Hidden);

static cl::opt<bool>
    ClCompat("stack-tagging-compat", cl::Hidden, cl::init(false),
             cl::ZeroOrMore,
             cl::desc("backwards compatible stack tagging: generate code "
                      "that runs on older hardware"));

static cl::opt<int> ClBisectStart("stack-tagging-bisect-start",
                                     cl::init(-1), cl::Hidden);
static cl::opt<int> ClBisectEnd("stack-tagging-bisect-end",
                                     cl::init(-1), cl::Hidden);
static cl::opt<bool> ClBisectDump("stack-tagging-bisect-dump",
                                     cl::init(false), cl::Hidden);

static const Align kTagGranuleSize = Align(16);

namespace {

class InitializerBuilder {
  uint64_t Size;
  const DataLayout *DL;
  Value *BasePtr;
  Function *SetTagFn;
  Function *SetTagZeroFn;
  Function *StgpFn;

  // List of initializers sorted by start offset.
  struct Range {
    uint64_t Start, End;
    Instruction *Inst;
  };
  SmallVector<Range, 4> Ranges;
  // 8-aligned offset => 8-byte initializer
  // Missing keys are zero initialized.
  std::map<uint64_t, Value *> Out;

public:
  InitializerBuilder(uint64_t Size, const DataLayout *DL, Value *BasePtr,
                     Function *SetTagFn, Function *SetTagZeroFn,
                     Function *StgpFn)
      : Size(Size), DL(DL), BasePtr(BasePtr), SetTagFn(SetTagFn),
        SetTagZeroFn(SetTagZeroFn), StgpFn(StgpFn) {}

  bool addRange(uint64_t Start, uint64_t End, Instruction *Inst) {
    auto I = std::lower_bound(
        Ranges.begin(), Ranges.end(), Start,
        [](const Range &LHS, uint64_t RHS) { return LHS.End <= RHS; });
    if (I != Ranges.end() && End > I->Start) {
      // Overlap - bail.
      return false;
    }
    Ranges.insert(I, {Start, End, Inst});
    return true;
  }

  bool addStore(uint64_t Offset, StoreInst *SI, const DataLayout *DL) {
    int64_t StoreSize = DL->getTypeStoreSize(SI->getOperand(0)->getType());
    if (!addRange(Offset, Offset + StoreSize, SI))
      return false;
    IRBuilder<> IRB(SI);
    applyStore(IRB, Offset, Offset + StoreSize, SI->getOperand(0));
    return true;
  }

  bool addMemSet(uint64_t Offset, MemSetInst *MSI) {
    uint64_t StoreSize = cast<ConstantInt>(MSI->getLength())->getZExtValue();
    if (!addRange(Offset, Offset + StoreSize, MSI))
      return false;
    IRBuilder<> IRB(MSI);
    applyMemSet(IRB, Offset, Offset + StoreSize,
                cast<ConstantInt>(MSI->getValue()));
    return true;
  }

  void applyMemSet(IRBuilder<> &IRB, int64_t Start, int64_t End,
                   ConstantInt *V) {
    // Out[] does not distinguish between zero and undef, and we already know
    // that this memset does not overlap with any other initializer. Nothing to
    // do for memset(0).
    if (V->isZero())
      return;
    for (int64_t Offset = Start - Start % 8; Offset < End; Offset += 8) {
      uint64_t Cst = 0x0101010101010101UL;
      int LowBits = Offset < Start ? (Start - Offset) * 8 : 0;
      if (LowBits)
        Cst = (Cst >> LowBits) << LowBits;
      int HighBits = End - Offset < 8 ? (8 - (End - Offset)) * 8 : 0;
      if (HighBits)
        Cst = (Cst << HighBits) >> HighBits;
      ConstantInt *C =
          ConstantInt::get(IRB.getInt64Ty(), Cst * V->getZExtValue());

      Value *&CurrentV = Out[Offset];
      if (!CurrentV) {
        CurrentV = C;
      } else {
        CurrentV = IRB.CreateOr(CurrentV, C);
      }
    }
  }

  // Take a 64-bit slice of the value starting at the given offset (in bytes).
  // Offset can be negative. Pad with zeroes on both sides when necessary.
  Value *sliceValue(IRBuilder<> &IRB, Value *V, int64_t Offset) {
    if (Offset > 0) {
      V = IRB.CreateLShr(V, Offset * 8);
      V = IRB.CreateZExtOrTrunc(V, IRB.getInt64Ty());
    } else if (Offset < 0) {
      V = IRB.CreateZExtOrTrunc(V, IRB.getInt64Ty());
      V = IRB.CreateShl(V, -Offset * 8);
    } else {
      V = IRB.CreateZExtOrTrunc(V, IRB.getInt64Ty());
    }
    return V;
  }

  void applyStore(IRBuilder<> &IRB, int64_t Start, int64_t End,
                  Value *StoredValue) {
    StoredValue = flatten(IRB, StoredValue);
    for (int64_t Offset = Start - Start % 8; Offset < End; Offset += 8) {
      Value *V = sliceValue(IRB, StoredValue, Offset - Start);
      Value *&CurrentV = Out[Offset];
      if (!CurrentV) {
        CurrentV = V;
      } else {
        CurrentV = IRB.CreateOr(CurrentV, V);
      }
    }
  }

  void generate(IRBuilder<> &IRB) {
    LLVM_DEBUG(dbgs() << "Combined initializer\n");
    // No initializers => the entire allocation is undef.
    if (Ranges.empty()) {
      emitUndef(IRB, 0, Size);
      return;
    }

    // Look through 8-byte initializer list 16 bytes at a time;
    // If one of the two 8-byte halfs is non-zero non-undef, emit STGP.
    // Otherwise, emit zeroes up to next available item.
    uint64_t LastOffset = 0;
    for (uint64_t Offset = 0; Offset < Size; Offset += 16) {
      auto I1 = Out.find(Offset);
      auto I2 = Out.find(Offset + 8);
      if (I1 == Out.end() && I2 == Out.end())
        continue;

      if (Offset > LastOffset)
        emitZeroes(IRB, LastOffset, Offset - LastOffset);

      Value *Store1 = I1 == Out.end() ? Constant::getNullValue(IRB.getInt64Ty())
                                      : I1->second;
      Value *Store2 = I2 == Out.end() ? Constant::getNullValue(IRB.getInt64Ty())
                                      : I2->second;
      emitPair(IRB, Offset, Store1, Store2);
      LastOffset = Offset + 16;
    }

    // memset(0) does not update Out[], therefore the tail can be either undef
    // or zero.
    if (LastOffset < Size)
      emitZeroes(IRB, LastOffset, Size - LastOffset);

    for (const auto &R : Ranges) {
      R.Inst->eraseFromParent();
    }
  }

  void emitZeroes(IRBuilder<> &IRB, uint64_t Offset, uint64_t Size) {
    LLVM_DEBUG(dbgs() << "  [" << Offset << ", " << Offset + Size
                      << ") zero\n");
    Value *Ptr = BasePtr;
    if (Offset)
      Ptr = IRB.CreateConstGEP1_32(Ptr, Offset);
    IRB.CreateCall(SetTagZeroFn,
                   {Ptr, ConstantInt::get(IRB.getInt64Ty(), Size)});
  }

  void emitUndef(IRBuilder<> &IRB, uint64_t Offset, uint64_t Size) {
    LLVM_DEBUG(dbgs() << "  [" << Offset << ", " << Offset + Size
                      << ") undef\n");
    Value *Ptr = BasePtr;
    if (Offset)
      Ptr = IRB.CreateConstGEP1_32(Ptr, Offset);
    IRB.CreateCall(SetTagFn, {Ptr, ConstantInt::get(IRB.getInt64Ty(), Size)});
  }

  void emitPair(IRBuilder<> &IRB, uint64_t Offset, Value *A, Value *B) {
    LLVM_DEBUG(dbgs() << "  [" << Offset << ", " << Offset + 16 << "):\n");
    LLVM_DEBUG(dbgs() << "    " << *A << "\n    " << *B << "\n");
    Value *Ptr = BasePtr;
    if (Offset)
      Ptr = IRB.CreateConstGEP1_32(Ptr, Offset);
    IRB.CreateCall(StgpFn, {Ptr, A, B});
  }

  Value *flatten(IRBuilder<> &IRB, Value *V) {
    if (V->getType()->isIntegerTy())
      return V;
    // vector of pointers -> vector of ints
    if (VectorType *VecTy = dyn_cast<VectorType>(V->getType())) {
      LLVMContext &Ctx = IRB.getContext();
      Type *EltTy = VecTy->getElementType();
      if (EltTy->isPointerTy()) {
        uint32_t EltSize = DL->getTypeSizeInBits(EltTy);
        Type *NewTy = VectorType::get(IntegerType::get(Ctx, EltSize),
                                      VecTy->getNumElements());
        V = IRB.CreatePointerCast(V, NewTy);
      }
    }
    return IRB.CreateBitOrPointerCast(
        V, IRB.getIntNTy(DL->getTypeStoreSize(V->getType()) * 8));
  }
};

class AArch64StackTagging : public FunctionPass {
  // This struct describes how and where an alloca needs to be tagged.
  struct AllocaInfo {
    AllocaInst *AI = nullptr;
    // Size of the tagged area, alloca size aligned up to 16.
    uint64_t Size = 0;
    // Allocation & address tag: [1, 15], and -1 for non-tagged allocations.
    int Tag = -1;
    // Allocas are extended to be a multiple of 16 bytes. This is either a
    // bitcast back to the original alloca type, or the original alloca itself
    // (if extension was not required). If AllocaBitCast != AI, then
    // AllocaBitCast is the only use of AI.
    Instruction *AllocaBitCast = nullptr;
    // All lifetime.start and lifetime.end calls for this alloc.
    SmallVector<IntrinsicInst *, 2> LifetimeStart;
    SmallVector<IntrinsicInst *, 2> LifetimeEnd;
    // All dbg.variable calls for this alloca.
    SmallVector<DbgVariableIntrinsic *, 2> DbgVariableIntrinsics;
    // This is where we want to insert address tagging (RAU of AllocaBitCast).
    // This location needs to dominate all uses of AllocaBitCast, with the
    // possible exception of a lifetime.start intrinsic or a bitcast whose only
    // use is the lifetime.start intrinsic.
    Instruction *TagAddressAfter = nullptr;
    // This is where we want to insert memory tagging. Must be dominated by
    // TagAddressAfter.
    Instruction *TagMemoryAfter = nullptr;
    // The list of places where we want to insert memory untagging. Either
    // immediately before a single lifetime.end, or at all function exits.
    SmallVector<Instruction *, 2> UntagBefore;
    // "true" here means that this alloca satisfies the conditions for
    // groupLifetimeStarts():
    // * TagAddressAfter == TagMemoryAfter == LifetimeStart[0]
    // * LifetimeStart.size() == 1
    // * lifetime.start dominates all uses of AllocaBitCast other than it's own
    //   operand, and possibly another bitcast inbetween (i.e. lifetime.start
    //   (bitcast (AllocaBitCast (alloca))) ).
    // * both bitcasts mentioned above immediately follow the alloca.
    // The single lifetime.start can then be hoisted within its basic block with
    // the goal of placing it adjacent to some other lifetime.start call and
    // then merging their conditional tagging basic blocks into one.
    bool CanHoistLifetimeStart = false;
    bool IsInteresting = false;
  };

  bool MergeInit;

public:
  static char ID; // Pass ID, replacement for typeid

  AArch64StackTagging(bool MergeInit = true)
      : FunctionPass(ID),
        MergeInit(ClMergeInit.getNumOccurrences() > 0 ? ClMergeInit
                                                      : MergeInit) {
    initializeAArch64StackTaggingPass(*PassRegistry::getPassRegistry());
  }

  bool isInterestingAlloca(const AllocaInst &AI);
  void alignAndPadAlloca(AllocaInfo &Info);

  void tagAlloca(AllocaInst *AI, Instruction *InsertBefore, Value *Ptr,
                 uint64_t Size);
  void untagAlloca(AllocaInst *AI, Instruction *InsertBefore, uint64_t Size);

  void splitLifetimeUses(AllocaInfo &Info, DominatorTree *DT);
  void groupLifetimeStarts(MapVector<AllocaInst *, AllocaInfo> &Allocas);

  Instruction *createTaggingBasicBlock(Instruction *InsertBefore, Value *Cond,
                                       const Twine &Name);
  void compatTagUntagAllocas(MapVector<AllocaInst *, AllocaInfo> &Allocas,
                             Value *CompatCond);
  void tagUntagAllocas(MapVector<AllocaInst *, AllocaInfo> &Allocas,
                       Instruction *BaseTaggedPointer);
  Instruction *collectInitializers(Instruction *StartInst, Value *StartPtr,
                                   uint64_t Size, InitializerBuilder &IB);

  BasicBlock *
  findPrologueBasicBlock(const MapVector<AllocaInst *, AllocaInfo> &Allocas,
                         const DominatorTree *DT);

  Instruction *insertBaseTaggedPointer(BasicBlock *PrologueBB);
  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override { return "AArch64 Stack Tagging"; }

private:
  Function *F;
  Module *M;
  Function *SetTagFunc;
  const DataLayout *DL;
  AAResults *AA;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    if (!ClCompat)
      AU.setPreservesCFG();
    if (MergeInit)
      AU.addRequired<AAResultsWrapperPass>();
  }
};

} // end anonymous namespace

char AArch64StackTagging::ID = 0;

INITIALIZE_PASS_BEGIN(AArch64StackTagging, DEBUG_TYPE, "AArch64 Stack Tagging",
                      false, false)
INITIALIZE_PASS_END(AArch64StackTagging, DEBUG_TYPE, "AArch64 Stack Tagging",
                    false, false)

FunctionPass *llvm::createAArch64StackTaggingPass(bool MergeInit) {
  return new AArch64StackTagging(MergeInit);
}

Instruction *AArch64StackTagging::collectInitializers(Instruction *StartInst,
                                                      Value *StartPtr,
                                                      uint64_t Size,
                                                      InitializerBuilder &IB) {
  MemoryLocation AllocaLoc{StartPtr, Size};
  Instruction *LastInst = StartInst;
  BasicBlock::iterator BI(StartInst);

  unsigned Count = 0;
  for (; Count < ClScanLimit && !BI->isTerminator(); ++BI) {
    if (!isa<DbgInfoIntrinsic>(*BI))
      ++Count;

    if (isNoModRef(AA->getModRefInfo(&*BI, AllocaLoc)))
      continue;

    if (!isa<StoreInst>(BI) && !isa<MemSetInst>(BI)) {
      // If the instruction is readnone, ignore it, otherwise bail out.  We
      // don't even allow readonly here because we don't want something like:
      // A[1] = 2; strlen(A); A[2] = 2; -> memcpy(A, ...); strlen(A).
      if (BI->mayWriteToMemory() || BI->mayReadFromMemory())
        break;
      continue;
    }

    if (StoreInst *NextStore = dyn_cast<StoreInst>(BI)) {
      if (!NextStore->isSimple())
        break;

      // Check to see if this store is to a constant offset from the start ptr.
      Optional<int64_t> Offset =
          isPointerOffset(StartPtr, NextStore->getPointerOperand(), *DL);
      if (!Offset)
        break;

      if (!IB.addStore(*Offset, NextStore, DL))
        break;
      LastInst = NextStore;
    } else {
      MemSetInst *MSI = cast<MemSetInst>(BI);

      if (MSI->isVolatile() || !isa<ConstantInt>(MSI->getLength()))
        break;

      if (!isa<ConstantInt>(MSI->getValue()))
        break;

      // Check to see if this store is to a constant offset from the start ptr.
      Optional<int64_t> Offset = isPointerOffset(StartPtr, MSI->getDest(), *DL);
      if (!Offset)
        break;

      if (!IB.addMemSet(*Offset, MSI))
        break;
      LastInst = MSI;
    }
  }
  return LastInst;
}

bool AArch64StackTagging::isInterestingAlloca(const AllocaInst &AI) {
  // FIXME: support dynamic allocas
  bool IsInteresting =
      AI.getAllocatedType()->isSized() && AI.isStaticAlloca() &&
      // alloca() may be called with 0 size, ignore it.
      AI.getAllocationSizeInBits(*DL).getValue() > 0 &&
      // inalloca allocas are not treated as static, and we don't want
      // dynamic alloca instrumentation for them as well.
      !AI.isUsedWithInAlloca() &&
      // swifterror allocas are register promoted by ISel
      !AI.isSwiftError() &&
      // safe allocas are not interesting
      !AI.getMetadata("stack-safe");
  return IsInteresting;
}

void AArch64StackTagging::tagAlloca(AllocaInst *AI, Instruction *InsertBefore,
                                    Value *Ptr, uint64_t Size) {
  auto SetTagZeroFunc =
      Intrinsic::getDeclaration(F->getParent(), Intrinsic::aarch64_settag_zero);
  auto StgpFunc =
      Intrinsic::getDeclaration(F->getParent(), Intrinsic::aarch64_stgp);

  InitializerBuilder IB(Size, DL, Ptr, SetTagFunc, SetTagZeroFunc, StgpFunc);
  bool LittleEndian =
      Triple(AI->getModule()->getTargetTriple()).isLittleEndian();
  // Current implementation of initializer merging assumes little endianness.
  if (MergeInit && !F->hasOptNone() && LittleEndian) {
    LLVM_DEBUG(dbgs() << "collecting initializers for " << *AI
                      << ", size = " << Size << "\n");
    InsertBefore = collectInitializers(InsertBefore, Ptr, Size, IB);
  }

  IRBuilder<> IRB(InsertBefore);
  IB.generate(IRB);
}

void AArch64StackTagging::untagAlloca(AllocaInst *AI, Instruction *InsertBefore,
                                      uint64_t Size) {
  IRBuilder<> IRB(InsertBefore);
  IRB.CreateCall(SetTagFunc, {IRB.CreatePointerCast(AI, IRB.getInt8PtrTy()),
                              ConstantInt::get(IRB.getInt64Ty(), Size)});
}

BasicBlock *AArch64StackTagging::findPrologueBasicBlock(
    const MapVector<AllocaInst *, AllocaInfo> &Allocas,
    const DominatorTree *DT) {
  BasicBlock *PrologueBB = nullptr;
  // Try sinking IRG as deep as possible to avoid hurting shrink wrap.
  for (auto &I : Allocas) {
    const AllocaInfo &Info = I.second;
    AllocaInst *AI = Info.AI;
    if (!Info.IsInteresting)
      continue;
    if (!PrologueBB) {
      PrologueBB = AI->getParent();
      continue;
    }
    PrologueBB = DT->findNearestCommonDominator(PrologueBB, AI->getParent());
  }
  assert(PrologueBB);
  return PrologueBB;
}

Instruction *
AArch64StackTagging::insertBaseTaggedPointer(BasicBlock *PrologueBB) {
  IRBuilder<> IRB(&PrologueBB->front());
  Function *IRG_SP =
      Intrinsic::getDeclaration(F->getParent(), Intrinsic::aarch64_irg_sp);
  Instruction *Base =
      IRB.CreateCall(IRG_SP, {Constant::getNullValue(IRB.getInt64Ty())});
  Base->setName("basetag");
  return Base;
}

void AArch64StackTagging::alignAndPadAlloca(AllocaInfo &Info) {
  const Align NewAlignment =
      max(MaybeAlign(Info.AI->getAlignment()), kTagGranuleSize);
  Info.AI->setAlignment(NewAlignment);

  uint64_t Size = Info.AI->getAllocationSizeInBits(*DL).getValue() / 8;
  uint64_t AlignedSize = alignTo(Size, kTagGranuleSize);
  if (Size == AlignedSize)
    return;

  // Add padding to the alloca.
  Type *AllocatedType =
      Info.AI->isArrayAllocation()
          ? ArrayType::get(
                Info.AI->getAllocatedType(),
                cast<ConstantInt>(Info.AI->getArraySize())->getZExtValue())
          : Info.AI->getAllocatedType();
  Type *PaddingType =
      ArrayType::get(Type::getInt8Ty(F->getContext()), AlignedSize - Size);
  Type *TypeWithPadding = StructType::get(AllocatedType, PaddingType);
  auto *NewAI = new AllocaInst(
      TypeWithPadding, Info.AI->getType()->getAddressSpace(), nullptr, "", Info.AI);
  NewAI->takeName(Info.AI);
  NewAI->setAlignment(MaybeAlign(Info.AI->getAlignment()));
  NewAI->setUsedWithInAlloca(Info.AI->isUsedWithInAlloca());
  NewAI->setSwiftError(Info.AI->isSwiftError());
  NewAI->copyMetadata(*Info.AI);

  auto *NewPtr = new BitCastInst(NewAI, Info.AI->getType(), "", Info.AI);
  Info.AI->replaceAllUsesWith(NewPtr);
  Info.AI->eraseFromParent();
  Info.AI = NewAI;
  Info.AllocaBitCast = NewPtr;
}

// Helper function to check for post-dominance.
static bool postDominates(const PostDominatorTree *PDT, const IntrinsicInst *A,
                          const IntrinsicInst *B) {
  const BasicBlock *ABB = A->getParent();
  const BasicBlock *BBB = B->getParent();

  if (ABB != BBB)
    return PDT->dominates(ABB, BBB);

  for (const Instruction &I : *ABB) {
    if (&I == B)
      return true;
    if (&I == A)
      return false;
  }
  llvm_unreachable("Corrupt instruction list");
}

void AArch64StackTagging::splitLifetimeUses(AllocaInfo &Info, DominatorTree *DT) {
  IntrinsicInst *LifetimeStart = Info.LifetimeStart[0];
  BitCastInst *LifetimeBitCast = nullptr;
  // errs() << "trying split for alloca\n  " << *Info.AllocaBitCast
  //        << "lifetime:\n  " << *LifetimeStart << "\n";
  for (const Use &U : Info.AllocaBitCast->uses()) {
    auto I = cast<Instruction>(U.getUser());
    if (I == LifetimeStart || DT->dominates(LifetimeStart, I))
      continue;
    if (LifetimeBitCast) {
      // FIXME: try harder to sink stuff with all uses dominated by lifetime.start.
      // errs() << "split failed: " << *I << "\n";
      return; // already got the single allowed pre-lifetime bitcast instruction
    }
    auto *BI = dyn_cast<BitCastInst>(I);
    if (!BI) {
      // errs() << "split failed: " << *I << "\n";
      return;
    }
    for (const Use &U : BI->uses()) {
      auto I = cast<Instruction>(U.getUser());
      if (I == LifetimeStart || DT->dominates(LifetimeStart, I))
        continue;
      // errs() << "split failed (bitcast use): " << *I << "\n";
      return;
    }
    LifetimeBitCast = BI;
  }
  // Success. All uses of the alloca are either the lifetime.start call or
  // dominated by it. Maybe with the exception of a single bitcast instruction
  // which, in turn, has the same property.
  // If there is such bitcast instruction with more than one use, it needs to be
  // split into one that feeds into lifetime.start, and another that can be sunk
  // below the lifetime.start call. Then address tagging can be sunk below
  // lifetime.start as well.
  Info.TagAddressAfter = Info.TagMemoryAfter;
  Info.CanHoistLifetimeStart = true;

  if (!LifetimeBitCast)
    return;

  if (LifetimeBitCast->hasOneUse()) {
    LifetimeBitCast->moveAfter(Info.AllocaBitCast);
  } else {
    Instruction *NewBitCast = LifetimeBitCast->clone();
    NewBitCast->insertAfter(Info.AllocaBitCast);
    LifetimeStart->setArgOperand(1, NewBitCast);
    LifetimeBitCast->moveAfter(LifetimeStart);
  }
}

void AArch64StackTagging::groupLifetimeStarts(
    MapVector<AllocaInst *, AllocaInfo> &Allocas) {
  SmallSet<IntrinsicInst*, 8> Movable;
  for (auto &I : Allocas) {
    const AllocaInfo &Info = I.second;
    if (Info.CanHoistLifetimeStart)
      Movable.insert(Info.LifetimeStart[0]);
  }

  for (auto &BB : *F) {
    Instruction *MoveBefore = nullptr;
    for (BasicBlock::iterator IT = BB.begin(); IT != BB.end(); ++IT) {
      Instruction *I = &*IT;

      auto *II = dyn_cast<IntrinsicInst>(I);
      if (II && Movable.count(II)) {
        if (MoveBefore) {
          II->moveBefore(MoveBefore);
        } else {
          MoveBefore = II->getNextNode();
        }
      }
      if ((II && II->getIntrinsicID() == Intrinsic::lifetime_end) ||
          isa<AllocaInst>(I)) {
        MoveBefore = nullptr;
        continue;
      }
    }
  }

  for (auto &I : Allocas) {
    AllocaInfo &Info = I.second;
    if (!Info.IsInteresting)
      continue;
    if (Info.LifetimeStart.size() == 1 &&
        Info.TagAddressAfter == Info.LifetimeStart[0]) {
      while (1) {
        IntrinsicInst *Next = dyn_cast_or_null<IntrinsicInst>(
            Info.TagAddressAfter->getNextNode());
        if (!Next || Next->getIntrinsicID() != Intrinsic::lifetime_start)
          break;
        // errs() << "update tag after from\n  " << *Info.TagAddressAfter
        //        << "\nto\n  " << *Next << "\n";
        if (Info.TagMemoryAfter == Info.TagAddressAfter)
          Info.TagMemoryAfter = Next;
        Info.TagAddressAfter = Next;
      }
    }

    for (Instruction *&UntagBefore : Info.UntagBefore) {
      while (1) {
        IntrinsicInst *Prev =
            dyn_cast_or_null<IntrinsicInst>(UntagBefore->getPrevNode());
        if (!Prev || Prev->getIntrinsicID() != Intrinsic::lifetime_end)
          break;
        // errs() << "update tag before from\n  " << *UntagBefore << "\nto\n  "
        //        << *Prev << "\n";
        UntagBefore = Prev;
      }
    }
  }
}

Instruction *
AArch64StackTagging::createTaggingBasicBlock(Instruction *InsertBefore,
                                             Value *Cond, const Twine &Name) {
  std::string PrevName = std::string(InsertBefore->getParent()->getName());
  Instruction *ThenTerm = SplitBlockAndInsertIfThen(
      Cond, InsertBefore, false,
      MDBuilder(M->getContext()).createBranchWeights(100, 1));
  ThenTerm->getParent()->setName(Name);
  InsertBefore->getParent()->setName(PrevName + ".cont");
  return ThenTerm;
}

void AArch64StackTagging::compatTagUntagAllocas(
    MapVector<AllocaInst *, AllocaInfo> &Allocas, Value *CompatCond) {
  // FIXME: use the alloca address as CompatCond

  // Map of tagging location (TagAddressAfter / TagMemoryAfter) to the insertion
  // location within a compat-conditional basic block.
  DenseMap<Instruction *, Instruction *> TagInsertPoint;
  for (auto &I : Allocas) {
    const AllocaInfo &Info = I.second;
    if (!Info.IsInteresting)
      continue;

    // Replace alloca with tagp(alloca).
    Instruction *&TagAddressBefore = TagInsertPoint[Info.TagAddressAfter];
    if (!TagAddressBefore)
      TagAddressBefore = createTaggingBasicBlock(
          Info.TagAddressAfter->getNextNode(), CompatCond, "mte.tag");

    IRBuilder<> IRB(TagAddressBefore);
    Function *IRG =
        Intrinsic::getDeclaration(F->getParent(), Intrinsic::aarch64_irg);
    Instruction *TaggedI =
        IRB.CreateCall(IRG, {Constant::getNullValue(IRB.getInt8PtrTy()),
                             Constant::getNullValue(IRB.getInt64Ty())});
    Value *Tagged =
        IRB.CreatePointerCast(TaggedI, Info.AllocaBitCast->getType());
    // if (AI.hasName())
    //   Tagged->setName(AI->getName() + ".tagged");

    IRB.SetInsertPoint(&TaggedI->getParent()->getSingleSuccessor()->front());
    PHINode *AllocaPhi = IRB.CreatePHI(Info.AllocaBitCast->getType(), 2);

    // Replace all uses with the tagged pointer. Skip the bitcast that feeds
    // into lifetime.start to avoid a circular reference.
    Info.AllocaBitCast->replaceUsesWithIf(AllocaPhi, [&](Use &U) {
      if (Info.LifetimeStart.empty())
        return true;
      Value *L = Info.LifetimeStart[0];
      Value *V = U.getUser();
      bool UsedInLifetimeStart =
          V == L || (V->hasOneUse() && V->user_back() == L);
      return !UsedInLifetimeStart;
    });

    AllocaPhi->addIncoming(Info.AllocaBitCast,
                           Info.TagAddressAfter->getParent());
    AllocaPhi->addIncoming(Tagged, TaggedI->getParent());
    if (Info.AllocaBitCast->getType() != IRB.getInt8PtrTy()) {
      TaggedI->setOperand(0, CastInst::CreatePointerCast(Info.AllocaBitCast,
                                                         IRB.getInt8PtrTy(), "",
                                                         TaggedI));
    } else {
      TaggedI->setOperand(0, Info.AllocaBitCast);
    }

    Instruction *&TagMemoryBefore = TagInsertPoint[Info.TagMemoryAfter];
    if (!TagMemoryBefore)
      TagMemoryBefore = createTaggingBasicBlock(
          Info.TagMemoryAfter->getNextNode(), CompatCond, "mte.tag");

    IRB.SetInsertPoint(TagMemoryBefore);
    Value *TaggedPtr;
    if (TagAddressBefore->getParent() == TagMemoryBefore->getParent()) {
      TaggedPtr = TaggedI;
    } else {
      TaggedPtr = IRB.CreatePointerCast(AllocaPhi, IRB.getInt8PtrTy());
    }
    tagAlloca(Info.AI, TagMemoryBefore, TaggedPtr, Info.Size);

    for (Instruction *I : Info.UntagBefore) {
      Instruction *&UntagBefore = TagInsertPoint[I];
      if (!UntagBefore)
        UntagBefore = createTaggingBasicBlock(I, CompatCond, "mte.untag");
      untagAlloca(Info.AI, UntagBefore, Info.Size);
    }
  }
}

void AArch64StackTagging::tagUntagAllocas(
    MapVector<AllocaInst *, AllocaInfo> &Allocas,
    Instruction *BaseTaggedPointer) {
  for (auto &I : Allocas) {
    const AllocaInfo &Info = I.second;
    AllocaInst *AI = Info.AI;
    if (!Info.IsInteresting)
      continue;

    Instruction *TagAddressBefore = Info.TagAddressAfter->getNextNode();
    Instruction *TagMemoryBefore = Info.TagMemoryAfter->getNextNode();

    IRBuilder<> IRB(TagAddressBefore);
    Function *TagP =
        Intrinsic::getDeclaration(F->getParent(), Intrinsic::aarch64_tagp,
                                  {Info.AllocaBitCast->getType()});
    Instruction *TagPCall = IRB.CreateCall(
        TagP,
        {Constant::getNullValue(Info.AllocaBitCast->getType()),
         BaseTaggedPointer, ConstantInt::get(IRB.getInt64Ty(), Info.Tag)});
    if (Info.AI->hasName())
      TagPCall->setName(Info.AI->getName() + ".tag");
    Info.AllocaBitCast->replaceAllUsesWith(TagPCall);
    TagPCall->setOperand(0, Info.AllocaBitCast);
    tagAlloca(AI, TagMemoryBefore,
              IRB.CreatePointerCast(TagPCall, IRB.getInt8PtrTy()), Info.Size);

    for (Instruction *UntagBefore : Info.UntagBefore)
      untagAlloca(AI, UntagBefore, Info.Size);
  }
}

// FIXME: check for MTE extension
bool AArch64StackTagging::runOnFunction(Function &Fn) {
  if (!Fn.hasFnAttribute(Attribute::SanitizeMemTag))
    return false;

  F = &Fn;
  M = F->getParent();
  DL = &Fn.getParent()->getDataLayout();
  if (MergeInit)
    AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();

  // F->dump();
  StringRef Name = F->getName();
  int Hash = llvm::MD5Hash(Name) % 1000000;
  if (ClBisectStart >= 0 && Hash < ClBisectStart)
    return false;
  if (ClBisectEnd >= 0 && Hash >= ClBisectEnd)
    return false;
  if (ClBisectDump)
    errs() << "ZZZ instrumenting: " << Name << "\n";

  MapVector<AllocaInst *, AllocaInfo> Allocas; // need stable iteration order
  SmallVector<Instruction *, 8> RetVec;
  DenseMap<Value *, AllocaInst *> AllocaForValue;
  SmallVector<Instruction *, 4> UnrecognizedLifetimes;

  for (auto &BB : *F) {
    for (BasicBlock::iterator IT = BB.begin(); IT != BB.end(); ++IT) {
      Instruction *I = &*IT;
      if (auto *AI = dyn_cast<AllocaInst>(I)) {
        Allocas[AI].AI = AI;
        Allocas[AI].AllocaBitCast = AI;
        continue;
      }

      if (auto *DVI = dyn_cast<DbgVariableIntrinsic>(I)) {
        if (auto *AI =
                dyn_cast_or_null<AllocaInst>(DVI->getVariableLocation())) {
          Allocas[AI].DbgVariableIntrinsics.push_back(DVI);
        }
        continue;
      }

      auto *II = dyn_cast<IntrinsicInst>(I);
      if (II && (II->getIntrinsicID() == Intrinsic::lifetime_start ||
                 II->getIntrinsicID() == Intrinsic::lifetime_end)) {
        AllocaInst *AI =
            llvm::findAllocaForValue(II->getArgOperand(1), AllocaForValue);
        if (!AI) {
          UnrecognizedLifetimes.push_back(I);
          continue;
        }
        if (II->getIntrinsicID() == Intrinsic::lifetime_start)
          Allocas[AI].LifetimeStart.push_back(II);
        else
          Allocas[AI].LifetimeEnd.push_back(II);
      }

      if (isa<ReturnInst>(I) || isa<ResumeInst>(I) || isa<CleanupReturnInst>(I))
        RetVec.push_back(I);
    }
  }

  if (Allocas.empty())
    return false;

  int NextTag = 0;
  int NumInterestingAllocas = 0;
  for (auto &I : Allocas) {
    AllocaInfo &Info = I.second;
    assert(Info.AI);

    if (!isInterestingAlloca(*Info.AI))
      continue;

    Info.IsInteresting = true;
    NumInterestingAllocas++;

    alignAndPadAlloca(Info);
    Info.Tag = NextTag;
    NextTag = (NextTag + 1) % 16;
  }

  if (NumInterestingAllocas == 0)
    return true;

  std::unique_ptr<DominatorTree> DeleteDT;
  DominatorTree *DT = nullptr;
  if (auto *P = getAnalysisIfAvailable<DominatorTreeWrapperPass>())
    DT = &P->getDomTree();

  if (DT == nullptr && (NumInterestingAllocas > 1 ||
                        !F->hasFnAttribute(Attribute::OptimizeNone))) {
    DeleteDT = std::make_unique<DominatorTree>(*F);
    DT = DeleteDT.get();
  }

  std::unique_ptr<PostDominatorTree> DeletePDT;
  PostDominatorTree *PDT = nullptr;
  if (auto *P = getAnalysisIfAvailable<PostDominatorTreeWrapperPass>())
    PDT = &P->getPostDomTree();

  if (PDT == nullptr && !F->hasFnAttribute(Attribute::OptimizeNone)) {
    DeletePDT = std::make_unique<PostDominatorTree>(*F);
    PDT = DeletePDT.get();
  }

  SetTagFunc =
      Intrinsic::getDeclaration(F->getParent(), Intrinsic::aarch64_settag);

  BasicBlock *PrologueBB = findPrologueBasicBlock(Allocas, DT);

  Value *CompatCond = nullptr;
  if (ClCompat) {
    IRBuilder<> IRB(&PrologueBB->front());
    auto FrameAddressFn = Intrinsic::getDeclaration(
        M, Intrinsic::frameaddress,
        IRB.getInt8PtrTy(M->getDataLayout().getAllocaAddrSpace()));
    Value *FrameAddress = IRB.CreateCall(
        FrameAddressFn, {Constant::getNullValue(IRB.getInt32Ty())});
    Type *IntptrTy = IRB.getIntPtrTy(*DL);
    CompatCond = IRB.CreateICmpNE(
        IRB.CreateAnd(IRB.CreatePointerCast(FrameAddress, IntptrTy),
                      ConstantInt::get(IntptrTy, 1ull << 60)),
        Constant::getNullValue(IntptrTy));

    CompatCond->setName("MTE");
  }

  // Decide where to tag and untag each alloca.
  for (auto &I : Allocas) {
    AllocaInfo &Info = I.second;
    if (!Info.IsInteresting)
      continue;

    if (UnrecognizedLifetimes.empty() && Info.LifetimeStart.size() == 1 &&
        Info.LifetimeEnd.size() == 1) {
      IntrinsicInst *Start = Info.LifetimeStart[0];
      IntrinsicInst *End = Info.LifetimeEnd[0];
      uint64_t Size =
          dyn_cast<ConstantInt>(Start->getArgOperand(0))->getZExtValue();
      Info.Size = alignTo(Size, kTagGranuleSize);
      Info.TagAddressAfter = Info.AllocaBitCast;
      Info.TagMemoryAfter = Start;
      // We need to ensure that if we tag some object, we certainly untag it
      // before the function exits.
      if (PDT != nullptr && postDominates(PDT, End, Start)) {
        Info.UntagBefore.push_back(End);
      } else {
        SmallVector<Instruction *, 8> ReachableRetVec;
        unsigned NumCoveredExits = 0;
        for (auto &RI : RetVec) {
          if (!isPotentiallyReachable(Start, RI, nullptr, DT))
            continue;
          ReachableRetVec.push_back(RI);
          if (DT != nullptr && DT->dominates(End, RI))
            ++NumCoveredExits;
        }
        // If there's a mix of covered and non-covered exits, just put the untag
        // on exits, so we avoid the redundancy of untagging twice.
        if (NumCoveredExits == ReachableRetVec.size()) {
          Info.UntagBefore.push_back(End);
        } else {
          for (auto &RI : ReachableRetVec)
            Info.UntagBefore.push_back(RI);
          // We may have inserted untag outside of the lifetime interval.
          // Remove the lifetime end call for this alloca.
          End->eraseFromParent();
        }
      }
    } else {
      Info.Size = Info.AI->getAllocationSizeInBits(*DL).getValue() / 8;
      Info.TagAddressAfter = Info.AllocaBitCast;
      Info.TagMemoryAfter = Info.AllocaBitCast;
      for (auto &RI : RetVec)
        Info.UntagBefore.push_back(RI);
      // We may have inserted tag/untag outside of any lifetime interval.
      // Remove all lifetime intrinsics for this alloca.
      for (auto &II : Info.LifetimeStart)
        II->eraseFromParent();
      Info.LifetimeStart.clear();
      for (auto &II : Info.LifetimeEnd)
        II->eraseFromParent();
      Info.LifetimeEnd.clear();
    }
  }

  if (ClCompat) {
    // See if we can generate the tagged pointer _after_ lifetime.start, where
    // the pointer tagging code can be merged with memory tagging code in a
    // single conditional BB. This requires that all uses of the alloca are
    // dominated by the lifetime.start instruction, which may require some code
    // changes to achieve (consider that lifetime.start itself is a user of the
    // alloca!).
    for (auto &I : Allocas) {
      AllocaInfo &Info = I.second;
      if (!Info.IsInteresting || Info.LifetimeStart.empty())
        continue;
      splitLifetimeUses(Info, DT);
    }

    // errs() << "==== after split ====\n";
    // F->dump();
    // See if we can move the code around to tag multiple allocas at once.
    // Matters only in the compat mode.
    groupLifetimeStarts(Allocas);

    // errs() << "==== after group ====\n";
    // F->dump();

    // Generate tagp and memory tagging/untagging code.
    compatTagUntagAllocas(Allocas, CompatCond);
  } else {
    tagUntagAllocas(Allocas, insertBaseTaggedPointer(PrologueBB));
  }

  for (auto &I : Allocas) {
    const AllocaInfo &Info = I.second;
    if (!Info.IsInteresting)
      continue;
    // Fixup debug intrinsics to point to the new alloca.
    for (auto DVI : Info.DbgVariableIntrinsics)
      DVI->setArgOperand(
          0,
          MetadataAsValue::get(F->getContext(), LocalAsMetadata::get(Info.AI)));
  }

  // If we have instrumented at least one alloca, all unrecognized lifetime
  // instrinsics have to go.
  for (auto &I : UnrecognizedLifetimes)
    I->eraseFromParent();

  return true;
}
