//===- IRDLSSAOps.cpp - IRDL-SSA dialect ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSA.h"
#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSAAttributes.h"
#include "mlir/Support/LogicalResult.h"
#include <memory>

using namespace mlir;
using namespace mlir::irdlssa;

// Utils

DynamicTypeDefinition *findDynamicType(MLIRContext &ctx, StringRef type) {
  auto splitted = type.split('.');
  auto dialectName = splitted.first;
  auto typeName = splitted.second;

  auto dialect = ctx.getOrLoadDialect(dialectName);
  if (!dialect)
    return nullptr;

  auto extensibleDialect = llvm::dyn_cast<ExtensibleDialect>(dialect);
  if (!extensibleDialect)
    return nullptr;

  return extensibleDialect->lookupTypeDefinition(typeName);
}

TypeWrapper *findTypeWrapper(MLIRContext &ctx, StringRef type) {
  Dialect *irdlssaDialect = ctx.getLoadedDialect("irdlssa");
  assert(irdlssaDialect && "irdlssa is not registered");

  IRDLSSADialect *irdlssa = dyn_cast<IRDLSSADialect>(irdlssaDialect);
  assert(irdlssa && "irdlssa dialect is not IRDL-SSA");

  return irdlssa->getTypeWrapper(type);
}

Attribute
instanciateParamType(llvm::function_ref<InFlightDiagnostic()> emitError,
                     MLIRContext &ctx, ParamTypeAttrOrAnyAttr attr) {
  if (ParamTypeInstanceAttr typeDesc =
          attr.getAttr().dyn_cast<ParamTypeInstanceAttr>()) {
    auto typeName = typeDesc.getBase();

    SmallVector<Attribute> params;
    for (ParamTypeAttrOrAnyAttr param : typeDesc.getParams()) {
      auto result = instanciateParamType(emitError, ctx, param);
      if (!result) {
        return Attribute();
      }
      params.push_back(result);
    }

    if (DynamicTypeDefinition *type = findDynamicType(ctx, typeName)) {
      DynamicType instanciated =
          DynamicType::getChecked(emitError, type, params);
      if (instanciated)
        return TypeAttr::get(instanciated);
      else
        return Attribute();
    } else if (TypeWrapper *type = findTypeWrapper(ctx, typeName)) {
      Type instanciated = type->instanciate(emitError, params);
      if (instanciated)
        return TypeAttr::get(instanciated);
      else
        return Attribute();
    } else {
      emitError().append("type ", typeName, " is not declared at that point");
      return {};
    }
  } else {
    return attr.getAttr();
  }
}

// IsType constraint

llvm::Optional<std::unique_ptr<TypeConstraint>>
SSA_IsType::getVerifier(SmallVector<Value> const &valueToConstr) {
  auto attr = instanciateParamType([&]() { return this->emitError(); },
                                   *this->getContext(), this->type());

  if (!attr)
    return {};

  if (TypeAttr type = attr.dyn_cast<TypeAttr>()) {
    return {std::make_unique<IsTypeConstraint>(type.getValue())};
  } else {
    this->emitError().append("the provided attribute is not a type");
    return {};
  }
}

ConstraintDomain getDomainOf(ParamTypeAttrOrAnyAttr attr) {
  ConstraintDomain domain;

  if (ParamTypeInstanceAttr paramType =
          attr.getAttr().dyn_cast<ParamTypeInstanceAttr>()) {
    std::vector<ConstraintDomain> paramDomains;
    for (ParamTypeAttrOrAnyAttr param : paramType.getParams()) {
      paramDomains.push_back(std::move(getDomainOf(param)));
    }
    domain.insert(paramType.getBase(), std::move(paramDomains));
  } else if (TypeAttr type = attr.getAttr().dyn_cast<TypeAttr>()) {
    domain.insert(type.getValue());
  } else {
    assert(0 && "non-type type parameters are currently not supported");
  }

  return std::move(domain);
}

ConstraintDomain
SSA_IsType::abstract(ArrayRef<ConstraintDomain> const &domains) {
  return std::move(getDomainOf(this->type()));
}

// ParametricType constraint

llvm::Optional<std::unique_ptr<TypeConstraint>>
SSA_ParametricType::getVerifier(SmallVector<Value> const &valueToConstr) {
  SmallVector<size_t> constraints;
  for (Value arg : this->args()) {
    for (size_t i = 0; i < valueToConstr.size(); i++) {
      if (valueToConstr[i] == arg) {
        constraints.push_back(i);
        break;
      }
    }
  }

  auto typeName = this->type();
  if (DynamicTypeDefinition *type =
          findDynamicType(*this->getContext(), typeName)) {
    return {std::make_unique<DynParametricTypeConstraint>(
        type, std::move(constraints))};
  } else if (TypeWrapper *type =
                 findTypeWrapper(*this->getContext(), typeName)) {
    return {std::make_unique<ParametricTypeConstraint>(type,
                                                       std::move(constraints))};
  } else {
    this->emitError().append("type ", typeName,
                             " is not declared at that point");
    return {};
  }
}

ConstraintDomain
SSA_ParametricType::abstract(ArrayRef<ConstraintDomain> const &domains) {
  ConstraintDomain domain;
  std::vector<ConstraintDomain> paramDomains(domains.size());
  for (auto &opDomain : domains) {
    paramDomains.push_back(opDomain);
  }
  domain.insert(this->type(), std::move(paramDomains));
  return std::move(domain);
}

// AnyOf constraint

llvm::Optional<std::unique_ptr<TypeConstraint>>
SSA_AnyOf::getVerifier(SmallVector<Value> const &valueToConstr) {
  SmallVector<size_t> constraints;
  for (Value arg : this->args()) {
    for (size_t i = 0; i < valueToConstr.size(); i++) {
      if (valueToConstr[i] == arg) {
        constraints.push_back(i);
        break;
      }
    }
  }

  return {std::make_unique<AnyOfTypeConstraint>(constraints)};
}

ConstraintDomain
SSA_AnyOf::abstract(ArrayRef<ConstraintDomain> const &domains) {
  assert(domains.size() > 0 &&
         "any_of should have at least one constraint parameter");

  ConstraintDomain newDomain = domains[0];
  for (auto &domain : domains.slice(1)) {
    ConstraintDomain building;

    for (World const &world1 : newDomain.worlds) {
      for (World const &world2 : domain.worlds) {
        Optional<World> inter = world1.intersect(world2);
        if (inter.hasValue()) {
          World worldPick1 = *inter;
          worldPick1.insert()
          building.worlds.insert()
        }
      }
    }

    std::swap(newDomain, building);
  }
}

// AnyType constraint

llvm::Optional<std::unique_ptr<TypeConstraint>>
SSA_AnyType::getVerifier(SmallVector<Value> const &valueToConstr) {
  return {std::make_unique<AnyTypeConstraint>()};
}

ConstraintDomain
SSA_AnyType::abstract(ArrayRef<ConstraintDomain> const &domains) {
  return std::move(ConstraintDomain::oneWorld(World::unconstrained()));
}
