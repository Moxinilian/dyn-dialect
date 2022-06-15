//===- Domain.h - IRDL-SSA domain analyss -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL-SSA/Analysis/Domain.h"

#include <utility>

using namespace mlir;
using namespace irdlssa;

ConstraintDomain ConstraintDomain::empty() {
  return std::move(ConstraintDomain());
}

ConstraintDomain ConstraintDomain::any() {
  ConstraintDomain domain;
  domain.isAny = true;
  return std::move(domain);
}

void ConstraintDomain::insert(StringRef base,
                              std::vector<ConstraintDomain> params) {
  if (!this->isAny)
    this->parametricTypes.insert({base, std::move(params)});
}

void ConstraintDomain::insert(Type type) {
  if (!this->isAny)
    this->builtinTypes.insert(type);
}

void ConstraintDomain::join(ConstraintDomain const &other) {
  if (this->isAny || other.isAny) {
    this->isAny = true;
    return;
  }

  for (Type t : other.builtinTypes) {
    this->builtinTypes.insert(t);
  }

  for (auto &entry : other.parametricTypes) {
    auto parametricTypeIt = this->parametricTypes.find(entry.getKey());
    if (parametricTypeIt != this->parametricTypes.end()) {
      auto &params = parametricTypeIt->second;
      auto &paramsOther = entry.second;
      assert(params.size() == paramsOther.size() &&
             "inconsistent amount of type parameters");
      for (size_t i = 0; i < params.size(); i++) {
        params[i].join(paramsOther[i]);
      }
    } else {
      this->parametricTypes.insert({entry.getKey(), entry.getValue()});
    }
  }
}

void ConstraintDomain::intersect(ConstraintDomain const &other) {
  if (other.isAny)
    return;

  if (this->isAny) {
    ConstraintDomain otherCopy = other;
    std::swap(*this, otherCopy);
    return;
  }

  SmallVector<Type> builtinOutsideIntersection;
  for (Type t : this->builtinTypes) {
    if (!other.builtinTypes.contains(t)) {
      builtinOutsideIntersection.push_back(t);
    }
  }

  for (Type t : builtinOutsideIntersection) {
    this->builtinTypes.erase(t);
  }

  SmallVector<StringRef> parametricOutsideIntersection;
  for (auto &entry : this->parametricTypes) {
    auto parametricTypeIt = other.parametricTypes.find(entry.getKey());
    if (parametricTypeIt == this->parametricTypes.end()) {
      parametricOutsideIntersection.push_back(entry.getKey());
    } else {
      auto &params = entry.second;
      auto &paramsOther = parametricTypeIt->second;
      assert(params.size() == paramsOther.size() &&
             "inconsistent amount of type parameters");
      for (size_t i = 0; i < params.size(); i++) {
        params[i].intersect(paramsOther[i]);
      }
    }
  }

  for (StringRef t : parametricOutsideIntersection) {
    this->parametricTypes.erase(t);
  }
}

bool ConstraintDomain::subsetOf(ConstraintDomain const &other) const {
  if (other.isAny)
    return true;

  if (this->isAny)
    return false;

  for (Type subsetType : this->builtinTypes) {
    if (!this->builtinTypes.contains(subsetType)) {
      return false;
    }
  }

  for (auto &subsetEntry : this->parametricTypes) {
    auto parametricTypeIt = this->parametricTypes.find(subsetEntry.getKey());
    if (parametricTypeIt != this->parametricTypes.end()) {
      auto &paramsSubset = subsetEntry.second;
      auto &paramsOther = parametricTypeIt->second;
      assert(paramsSubset.size() == paramsOther.size() &&
             "inconsistent amount of type parameters");
      for (size_t i = 0; i < paramsSubset.size(); i++) {
        if (!paramsSubset[i].subsetOf(paramsOther[i])) {
          return false;
        }
      }
    } else {
      return false;
    }
  }

  return true;
}

Optional<size_t> ConstraintDomain::size() const {
  if (this->isAny)
    return llvm::None;

  size_t size = this->builtinTypes.size();
  
  for (auto &entry : this->parametricTypes) {
    size_t productDomainSize = 1;
    for (auto &param : entry.getValue()) {
      Optional<size_t> paramSize = param.size();
      if (paramSize.hasValue()) {
        productDomainSize *= *paramSize;
      } else {
        return llvm::None;
      }
    }
    size += productDomainSize;
  }

  return size;
}
