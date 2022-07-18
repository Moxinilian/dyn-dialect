//===- IRDLEvalOps.cpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL-Eval/IR/IRDLEval.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace irdlssa;
using namespace irdleval;

LogicalResult Verifier::verify() {
  Operation *parent = this->getOperation()->getParentOp();
  assert(parent && "verifier operation has no parent");

  size_t argAmount = 0;
  for (Operation &op : parent->getRegion(0).getOps()) {
    if (SSA_OperandsOp operOp = llvm::dyn_cast<SSA_OperandsOp>(op)) {
      argAmount += operOp.args().size();
    } else if (SSA_ResultsOp resOp = llvm::dyn_cast<SSA_ResultsOp>(op)) {
      argAmount += resOp.args().size();
    } else if (SSA_ParametersOp paramOp =
                   llvm::dyn_cast<SSA_ParametersOp>(op)) {
      argAmount += paramOp.args().size();
    }
  }

  if (this->body().getArguments().size() != argAmount) {
    return this->emitError().append("verifier body region expected ", argAmount,
                                    " arguments but got ",
                                    this->body().getArguments().size());
  }

  auto args = this->body().getArguments();
  for (size_t i = 0; i < args.size(); i++) {
    if (!args[i].getType().isa<EvalTypeType>()) {
      return this->emitError().append(
          "argument ", args[i],
          " of verifier body region is of incorrect type ", args[i].getType());
    }
  }

  return LogicalResult::success();
}

LogicalResult MatchType::verify() {
  if (this->success()->getArguments().size() != 0) {
    return this->emitError().append(
        "success block expected to have 0 arguments, has ",
        this->success()->getArguments().size());
  }

  if (this->failure()->getArguments().size() != 0) {
    return this->emitError().append(
        "failure block expected to have 0 arguments, has ",
        this->failure()->getArguments().size());
  }

  return LogicalResult::success();
}

LogicalResult CheckType::verify() {
  if (this->success()->getArguments().size() != 0) {
    return this->emitError().append(
        "success block expected to have 0 arguments, has ",
        this->success()->getArguments().size());
  }

  if (this->failure()->getArguments().size() != 0) {
    return this->emitError().append(
        "failure block expected to have 0 arguments, has ",
        this->failure()->getArguments().size());
  }

  return LogicalResult::success();
}

LogicalResult CheckParametric::verify() {
  auto args = this->success()->getArguments();
  for (size_t i = 0; i < args.size(); i++) {
    if (!args[i].getType().isa<EvalTypeType>()) {
      return this->emitError().append("success block argument ", args[i],
                                      " is of incorrect type ",
                                      args[i].getType());
    }
  }

  if (this->invalidBase()->getArguments().size() != 0) {
    return this->emitError().append(
        "invalid base block expected to have 0 arguments, has ",
        this->invalidBase()->getArguments().size());
  }

  return LogicalResult::success();
}

#define GET_OP_CLASSES
#include "Dyn/Dialect/IRDL-Eval/IR/IRDLEvalOps.cpp.inc"
