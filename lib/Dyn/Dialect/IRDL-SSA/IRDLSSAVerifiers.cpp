//===- IRDLSSAVerifiers.cpp - IRDL-SSA verifiers ------------------ C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Verifiers for objects declared by IRDL-SSA.
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL-SSA/IRDLSSAVerifiers.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace irdlssa {

ConstraintVerifier::ConstraintVerifier(
    llvm::ArrayRef<std::unique_ptr<TypeConstraint>> constraints)
    : constraints(constraints), assigned() {
  assigned.resize(this->constraints.size());
}

LogicalResult ConstraintVerifier::verifyType(
    Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
    size_t variable) {

  if (variable >= this->constraints.size()) {
    if (emitError)
      return (*emitError)().append("invalid constraint variable ", variable);
    else
      return failure();
  }

  if (this->assigned[variable].hasValue()) {
    if (type == this->assigned[variable].getValue())
      return success();
    else {
      if (emitError)
        return (*emitError)().append("expected type ",
                                     this->assigned[variable].getValue(),
                                     " but got ", type);
      else
        return failure();
    }
  }

  auto result = this->constraints[variable]->verifyType(emitError, type, *this);
  if (succeeded(result))
    this->assigned[variable] = type;

  return result;
}

LogicalResult IsTypeConstraint::verifyType(
    Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
    ConstraintVerifier &context) const {
  if (type == this->expectedType)
    return success();

  if (emitError)
    return (*emitError)().append("expected type ", this->expectedType,
                                 " but got type ", type);
  return failure();
}

LogicalResult ParametricTypeConstraint::verifyType(
    Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
    ConstraintVerifier &context) const {
  if (!this->expectedType->isCorrectType(type)) {
    if (emitError)
      return (*emitError)().append("expected base type '",
                                   this->expectedType->getName(),
                                   "' but got type ", type);
    return failure();
  }

  auto params = this->expectedType->getParameters(type);
  // Since we do not have variadic parameters yet, we should have the
  // exact number of constraints.
  // TODO: Handle this gracefully upstream
  assert(params.size() == this->constraints.size() &&
         "unexpected number of parameters in parameter type constraint");
  for (size_t i = 0; i < params.size(); i++) {
    auto paramType = params[i].cast<TypeAttr>().getValue();
    if (failed(context.verifyType(emitError, paramType, this->constraints[i])))
      return failure();
  }

  return success();
}

LogicalResult DynParametricTypeConstraint::verifyType(
    Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
    ConstraintVerifier &context) const {
  auto dynType = type.dyn_cast<DynamicType>();
  if (!dynType || dynType.getTypeDef() != this->expectedType) {
    if (emitError)
      return (*emitError)().append(
          "expected base type '",
          this->expectedType->getDialect()->getNamespace(), ".",
          this->expectedType->getName(), "' but got type ", type);
    return failure();
  }

  // Since we do not have variadic parameters yet, we should have the
  // exact number of constraints.
  // TODO: Handle this gracefully upstream
  assert(dynType.getParams().size() == this->constraints.size() &&
         "unexpected number of parameters in parameter type constraint");
  auto params = dynType.getParams();
  for (size_t i = 0; i < params.size(); i++) {
    auto paramType = params[i].cast<TypeAttr>().getValue();
    if (failed(context.verifyType(emitError, paramType, this->constraints[i])))
      return failure();
  }

  return success();
}

LogicalResult AnyOfTypeConstraint::verifyType(
    Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
    ConstraintVerifier &context) const {
  for (size_t constr : this->constraints) {
    if (succeeded(context.verifyType({}, type, constr))) {
      return success();
    }
  }

  if (emitError)
    return (*emitError)().append("type ", type,
                                 " does not satisfy the constraint");
  return failure();
}

LogicalResult AnyTypeConstraint::verifyType(
    Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
    ConstraintVerifier &context) const {
  return success();
}

} // namespace irdlssa
} // namespace mlir
