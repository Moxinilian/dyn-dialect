//===- Domain.h - IRDL-SSA domain analysis ----------------------*- C++ -*-===//
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

TypeSet TypeSet::empty() { return std::move(TypeSet()); }

TypeSet TypeSet::any() {
  TypeSet set;
  set.isAny = true;
  return std::move(set);
}

void TypeSet::insert(StringRef base, std::vector<TypeSet> params) {
  if (!this->isAny)
    this->parametricTypes.insert({base, std::move(params)});
}

void TypeSet::insert(Type type) {
  if (!this->isAny)
    this->builtinTypes.insert(type);
}

TypeSet TypeSet::join(TypeSet const &other) const {
  if (this->isAny || other.isAny) {
    return std::move(TypeSet::any());
  }

  llvm::StringMap<std::vector<TypeSet>> newParametricTypes;
  llvm::SmallPtrSet<Type, 4> newBuiltinTypes;

  for (Type t : this->builtinTypes) {
    newBuiltinTypes.insert(t);
  }

  for (Type t : other.builtinTypes) {
    newBuiltinTypes.insert(t);
  }

  for (auto &entry : other.parametricTypes) {
    auto parametricTypeIt = this->parametricTypes.find(entry.getKey());
    if (parametricTypeIt != this->parametricTypes.end()) {
      auto &params = parametricTypeIt->second;
      auto &paramsOther = entry.second;
      assert(params.size() == paramsOther.size() &&
             "inconsistent amount of type parameters");
      std::vector<TypeSet> newParamTypes(params.size());
      for (size_t i = 0; i < params.size(); i++) {
        newParamTypes.push_back(std::move(params[i].join(paramsOther[i])));
      }
      newParametricTypes.insert({entry.getKey(), std::move(newParamTypes)});
    } else {
      newParametricTypes.insert({entry.getKey(), entry.getValue()});
    }
  }

  for (auto &entry : this->parametricTypes) {
    if (newParametricTypes.count(entry.getKey()) == 0) {
      newParametricTypes.insert({entry.getKey(), entry.getValue()});
    }
  }

  return std::move(TypeSet(newParametricTypes, newBuiltinTypes));
}

TypeSet TypeSet::intersect(TypeSet const &other) const {
  if (other.isAny)
    return *this;

  if (this->isAny) {
    return other;
  }

  llvm::StringMap<std::vector<TypeSet>> newParametricTypes;
  llvm::SmallPtrSet<Type, 4> newBuiltinTypes;

  for (Type t : this->builtinTypes) {
    if (other.builtinTypes.count(t) > 0) {
      newBuiltinTypes.insert(t);
    }
  }

  for (auto &entry : this->parametricTypes) {
    auto parametricTypeIt = other.parametricTypes.find(entry.getKey());
    if (parametricTypeIt != this->parametricTypes.end()) {
      auto &params = entry.second;
      auto &paramsOther = parametricTypeIt->second;
      assert(params.size() == paramsOther.size() &&
             "inconsistent amount of type parameters");
      std::vector<TypeSet> newParamTypes(params.size());
      for (size_t i = 0; i < params.size(); i++) {
        newParamTypes.push_back(std::move(params[i].intersect(paramsOther[i])));
      }
      newParametricTypes.insert({entry.getKey(), std::move(newParamTypes)});
    }
  }

  return std::move(TypeSet(newParametricTypes, newBuiltinTypes));
}

bool TypeSet::subsetOf(TypeSet const &other) const {
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

bool TypeSet::isEmpty() const {
  if (this->isAny || this->builtinTypes.size() > 0)
    return false;

  for (auto &paramType : this->parametricTypes) {
    if (paramType.getValue().size() == 0) {
      // If the type has no parameters, then the base
      // alone testifies of a recognized type.
      return false;
    }

    for (TypeSet const &paramSet : paramType.getValue()) {
      if (!paramSet.isEmpty())
        return false;
    }
  }

  return true;
}

Optional<size_t> TypeSet::size() const {
  if (this->isAny)
    return llvm::None;

  size_t size = this->builtinTypes.size();

  for (auto &entry : this->parametricTypes) {
    size_t productTypeSetSize = 1;
    for (auto &param : entry.getValue()) {
      Optional<size_t> paramSize = param.size();
      if (paramSize.hasValue()) {
        productTypeSetSize *= *paramSize;
      } else {
        return llvm::None;
      }
    }
    size += productTypeSetSize;
  }

  return size;
}

World World::unconstrained() {
  return World();
}

void World::insert(Value val, TypeSet types) {
  this->typeSets.insert({val, std::move(types)});
}

Optional<World> World::intersect(World const &other) const {
  World world;

  for (auto &entry : this->typeSets) {
    auto otherSetIt = other.typeSets.find(entry.getFirst());
    if (otherSetIt != this->typeSets.end()) {
      auto &typeSet = entry.second;
      auto &otherTypeSet = otherSetIt->second;
      TypeSet result = typeSet.intersect(otherTypeSet);
      if (result.isEmpty()) {
        return llvm::None;
      }
      world.insert(entry.getFirst(), std::move(result));
    } else {
      world.insert(entry.getFirst(), entry.getSecond());
    }
  }

  for (auto &entry : other.typeSets) {
    if (world.typeSets.count(entry.getFirst()) == 0) {
      world.insert(entry.getFirst(), entry.getSecond());
    }
  }

  return std::move(world);
}

ConstraintDomain ConstraintDomain::oneWorld(World world) {
  ConstraintDomain domain;
  domain.worlds.insert(world);
  return std::move(domain);
}
