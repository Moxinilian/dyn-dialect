//===- DynOps.h - Dyn dialect ops -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DYNOPS_H
#define DYN_DYNOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

class DynOp : public mlir::Op<DynOp> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "dyn.dynop"; }
};

#endif // DYN_DYNOPS_H
