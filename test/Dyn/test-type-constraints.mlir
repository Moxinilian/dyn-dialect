// RUN: dyn-opt %s --irdl-file=%S/testd.irdl -split-input-file -verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// Equality constraint
//===----------------------------------------------------------------------===//

func @succeededEqConstraint() {
  // CHECK: "testd.eq"() : () -> i32
  "testd.eq"() : () -> i32
  return
}

// -----

func @failedEqConstraint() {
  // expected-error@+1 {{expected type 'i32' but got type 'i64'}}
  "testd.eq"() : () -> i64
  return
}

// -----

//===----------------------------------------------------------------------===//
// AnyOf constraint
//===----------------------------------------------------------------------===//

func @succeededAnyOfConstraint() {
  // CHECK: "testd.anyof"() : () -> i32
  "testd.anyof"() : () -> i32
  // CHECK: "testd.anyof"() : () -> i64
  "testd.anyof"() : () -> i64
  return
}

// -----

func @failedAnyOfConstraint() {
  // expected-error@+1 {{type 'i1' does not satisfy the constraint}}
  "testd.anyof"() : () -> i1
  return
}

// -----

//===----------------------------------------------------------------------===//
// Any constraint
//===----------------------------------------------------------------------===//

func @succeededAnyConstraint() {
  // CHECK: "testd.any"() : () -> i32
  "testd.any"() : () -> i32
  // CHECK: "testd.any"() : () -> i64
  "testd.any"() : () -> i64
  return
}

// -----

//===----------------------------------------------------------------------===//
// Non-dynamic parameters constraint
//===----------------------------------------------------------------------===//

func @succeededParamsConstraint() {
  // CHECK: "testd.params"() : () -> complex<i32>
  "testd.params"() : () -> complex<i32>
  // CHECK: "testd.params"() : () -> complex<i64>
  "testd.params"() : () -> complex<i64>
  return
}

// -----

func @failedDynParamsConstraintBase() {
  // expected-error@+1 {{expected base type 'std.complex' but got type 'i32'}}
  "testd.params"() : () -> i32
  return
}

// -----

func @failedDynParamsConstraintParam() {
  // expected-error@+1 {{type 'i1' does not satisfy the constraint}}
  "testd.params"() : () -> complex<i1>
  return
}

// -----

//===----------------------------------------------------------------------===//
// Dynamic base constraint
//===----------------------------------------------------------------------===//

func @succeededDynBaseConstraint() {
  // CHECK: "testd.dynbase"() : () -> !testd.parametric<i32>
  "testd.dynbase"() : () -> !testd.parametric<i32>
  // CHECK: "testd.dynbase"() : () -> !testd.parametric<i64>
  "testd.dynbase"() : () -> !testd.parametric<i64>
  // CHECK: "testd.dynbase"() : () -> !testd.parametric<!testd.parametric<i64>>
  "testd.dynbase"() : () -> !testd.parametric<!testd.parametric<i64>>
  return
}

// -----

func @failedDynBaseConstraint() {
  // expected-error@+1 {{expected base type 'testd.parametric' but got type 'i32'}}
  "testd.dynbase"() : () -> i32
  return
}

// -----

//===----------------------------------------------------------------------===//
// Dynamic parameters constraint
//===----------------------------------------------------------------------===//

func @succeededDynParamsConstraint() {
  // CHECK: "testd.dynparams"() : () -> !testd.parametric<i32>
  "testd.dynparams"() : () -> !testd.parametric<i32>
  // CHECK: "testd.dynparams"() : () -> !testd.parametric<i64>
  "testd.dynparams"() : () -> !testd.parametric<i64>
  return
}

// -----

func @failedDynParamsConstraintBase() {
  // expected-error@+1 {{expected base type 'testd.parametric' but got type 'i32'}}
  "testd.dynparams"() : () -> i32
  return
}

// -----

func @failedDynParamsConstraintParam() {
  // expected-error@+1 {{type 'i1' does not satisfy the constraint}}
  "testd.dynparams"() : () -> !testd.parametric<i1>
  return
}

// -----

//===----------------------------------------------------------------------===//
// Constraint variables
//===----------------------------------------------------------------------===//

func @succeededConstraintVars() {
  // CHECK: "testd.constraint_vars"() : () -> (i32, i32)
  "testd.constraint_vars"() : () -> (i32, i32)
  return
}

// -----

func @succeededConstraintVars2() {
  // CHECK: "testd.constraint_vars"() : () -> (i64, i64)
  "testd.constraint_vars"() : () -> (i64, i64)
  return
}

// -----

func @failedConstraintVars() {
  // expected-error@+1 {{expected 'i64' but got 'i32'}}
  "testd.constraint_vars"() : () -> (i64, i32)
  return
}
