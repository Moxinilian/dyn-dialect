//===- dyn-opt.cpp ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "MlirOptMain.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <type_traits>

using namespace mlir;
using namespace irdl;

class ComplexTypeWrapper : public ConcreteTypeWrapper<ComplexType> {
  StringRef getName() override { return "std.complex"; }

  SmallVector<Attribute> getParameters(ComplexType type) override {
    return {TypeAttr::get(type.getElementType())};
  }
};

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  // TODO: Register passes here.

  MLIRContext ctx;
  auto irdl = ctx.getOrLoadDialect<irdl::IRDLDialect>();

  irdl->addTypeWrapper<ComplexTypeWrapper>();

  // Register the standard dialect and the IRDL dialect in the MLIR context
  DialectRegistry registry;
  registry.insert<StandardOpsDialect>();
  ctx.appendDialectRegistry(registry);

  return failed(mlir::MlirOptMain(argc, argv, "Dyn optimizer driver\n", ctx));
}
