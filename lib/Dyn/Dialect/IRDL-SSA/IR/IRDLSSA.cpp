//===- IRDLSSA.cpp - IRDL-SSA dialect ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSA.h"
#include "Dyn/Dialect/IRDL-SSA/IRDLSSARegistration.h"
#include "Dyn/Dialect/IRDL-SSA/TypeWrapper.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::irdlssa;

using ArgDef = std::pair<StringRef, Attribute>;
using ArgDefs = ArrayRef<ArgDef>;

//===----------------------------------------------------------------------===//
// IRDL dialect.
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSA.cpp.inc"

#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSADialect.cpp.inc"

void IRDLSSADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSAOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSATypesGen.cpp.inc"
      >();
  // registerAttributes();
}

void IRDLSSADialect::addTypeWrapper(std::unique_ptr<TypeWrapper> wrapper) {
  this->irdlssaContext.addTypeWrapper(std::move(wrapper));
}

TypeWrapper *IRDLSSADialect::getTypeWrapper(StringRef typeName) {
  return this->irdlssaContext.getTypeWrapper(typeName);
}

//===----------------------------------------------------------------------===//
// Parsing/Printing
//===----------------------------------------------------------------------===//

static ParseResult parseKeywordOrString(OpAsmParser &p, StringAttr &attr) {
  std::string str;
  if (failed(p.parseKeywordOrString(&str)))
    return failure();
  attr = p.getBuilder().getStringAttr(str);
  return success();
}

static void printKeywordOrString(OpAsmPrinter &p, Operation *,
                                 StringAttr attr) {
  p.printKeywordOrString(attr.getValue());
}

/// Parse a region, and add a single block if the region is empty.
/// If no region is parsed, create a new region with a single empty block.
static ParseResult parseSingleBlockRegion(OpAsmParser &p, Region &region) {
  auto regionParseRes = p.parseOptionalRegion(region);
  if (regionParseRes.hasValue()) {
    if (failed(regionParseRes.getValue()))
      return failure();
  }
  // If the region is empty, add a single empty block.
  if (region.getBlocks().size() == 0) {
    region.push_back(new Block());
  }

  return success();
}

static void printSingleBlockRegion(OpAsmPrinter &p, Operation *op,
                                   Region &region) {
  if (!region.getBlocks().front().empty()) {
    p.printRegion(region);
  }
}

LogicalResult SSA_DialectOp::verify() {
  return success(Dialect::isValidNamespace(name()));
}

#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSAInterfaces.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSATypesGen.cpp.inc"

#define GET_OP_CLASSES
#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSAOps.cpp.inc"
