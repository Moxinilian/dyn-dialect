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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <vector>

namespace mlir {
namespace irdlssa {

class ConstraintDomain {
  // TODO: This is not really the most efficient memory layout.
  bool isAny = false;
  llvm::StringMap<std::vector<ConstraintDomain>> parametricTypes;
  llvm::SmallPtrSet<Type, 4> builtinTypes;


public:
  /// Constructs an empty domain.
  ConstraintDomain() {}

  static ConstraintDomain empty();
  static ConstraintDomain any();

  void insert(StringRef base, std::vector<ConstraintDomain> params);
  void insert(Type type);

  /// Adds to the current domain all types
  /// in the `other` domain.
  void join(ConstraintDomain const &other);

  /// Removes from the current domain all types
  /// that are not in the `other` domain.
  void intersect(ConstraintDomain const &other);

  /// Determines if the current domain is a subset
  /// of the provided `other` domain, that is that all
  /// types within this domain are also within the
  /// `other` domain.
  bool subsetOf(ConstraintDomain const &other) const;

  /// Returns the amount of different concrete types this domain
  /// contains. No size means it contains an infinite amount.
  Optional<size_t> size() const;
};

struct DomainAnalysis {
  ConstraintDomain domain;

  DomainAnalysis(Operation *op);
};

} // namespace irdlssa
} // namespace mlir

#endif // DYN_DIALECT_IRDL_SSA_ANALYSIS_DOMAIN_H_
