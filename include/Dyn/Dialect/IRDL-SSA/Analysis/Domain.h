//===- Domain.h - IRDL-SSA domain analyss -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Analysis to compute the possible types an SSA constraint can represent.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_SSA_ANALYSIS_DOMAIN_H_
#define DYN_DIALECT_IRDL_SSA_ANALYSIS_DOMAIN_H_

#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <vector>

namespace mlir {
namespace irdlssa {

// TODO: Make this more efficient

class TypeSet {
  bool isAny = false;
  llvm::StringMap<std::vector<TypeSet>> parametricTypes;
  llvm::SmallPtrSet<Type, 4> builtinTypes;

  TypeSet(llvm::StringMap<std::vector<TypeSet>> parametricTypes,
          llvm::SmallPtrSet<Type, 4> builtinTypes)
      : parametricTypes(parametricTypes), builtinTypes(builtinTypes) {}

public:
  /// Constructs an empty domain.
  TypeSet() {}

  static TypeSet empty();
  static TypeSet any();

  void insert(StringRef base, std::vector<TypeSet> params);
  void insert(Type type);

  /// Creates a type set containing all the types
  /// for the current set and the `other` set.
  TypeSet join(TypeSet const &other) const;

  /// Creates a type set containing the types that
  /// are both in the current set and the `other` set.
  TypeSet intersect(TypeSet const &other) const;

  /// Determines if the current type set is a subset
  /// of the provided `other` type set, that is that all
  /// types within this type set are also within the
  /// `other` type set.
  bool subsetOf(TypeSet const &other) const;

  bool isEmpty() const;

  /// Returns the amount of different concrete types this type set
  /// contains. No size means it contains an infinite amount.
  Optional<size_t> size() const;
};

class World {
  llvm::DenseMap<Value, TypeSet> typeSets;

public:
  static World unconstrained();

  void insert(Value val, TypeSet types);

  /// Intersects two worlds together to represent the values
  /// that are common to the two. Returns None if no such
  /// cases exist.
  Optional<World> intersect(World const &other) const;
};

struct ConstraintDomain {
  llvm::DenseSet<World> worlds;

  static ConstraintDomain oneWorld(World world);
};

} // namespace irdlssa
} // namespace mlir

#endif // DYN_DIALECT_IRDL_SSA_ANALYSIS_DOMAIN_H_
