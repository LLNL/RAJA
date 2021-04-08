//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing hip register allocation error reproducer.
///
/// Issue: Can not build certain code with rocm/4.1.0.
///
/// Seen when compiling for gfx906 on ElCap EA systems (AMD MI60)
///
/// NOTES:
///
/// lld produces this error message
///   error: ran out of registers during register allocation
///
/// This uses 40 atomics, but only 21 atomics are required to cause the error
///
/// Writing an equivalent kernel in native HIP does not reproduce the error
///

#include <cstdlib>
#include <iostream>
#include <iomanip>

#include "RAJA/RAJA.hpp"
#include "RAJA/policy/hip/raja_hiperrchk.hpp"

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{
  double* d;
  hipErrchk(hipHostMalloc((void **)&d, sizeof(double) * 1, hipHostMallocDefault));
  d[0] = 1.0;
  double* e;
  hipErrchk(hipHostMalloc((void **)&e, sizeof(double) * 1, hipHostMallocDefault));
  e[0] = 0.0;

  double* a_00 = d;
  double* b_00 = d;
  double* b_01 = d;
  double* b_02 = d;
  double* b_03 = d;
  double* b_04 = d;
  double* b_05 = d;
  double* b_06 = d;
  double* b_07 = d;
  double* b_08 = d;
  double* b_09 = d;
  double* b_10 = d;
  double* b_11 = d;
  double* b_12 = d;
  double* b_13 = d;
  double* b_14 = d;
  double* b_15 = d;
  double* b_16 = d;
  double* b_17 = d;
  double* b_18 = d;
  double* b_19 = d;
  double* b_20 = d;
  double* b_21 = d;
  double* b_22 = d;
  double* b_23 = d;
  double* b_24 = d;
  double* b_25 = d;
  double* b_26 = d;
  double* b_27 = d;
  double* b_28 = d;
  double* b_29 = d;
  double* b_30 = d;
  double* b_31 = d;
  double* b_32 = d;
  double* b_33 = d;
  double* b_34 = d;
  double* b_35 = d;
  double* b_36 = d;
  double* b_37 = d;
  double* b_38 = d;
  double* b_39 = d;
  double* c_00 = e;
  double* c_01 = e;
  double* c_02 = e;
  double* c_03 = e;
  double* c_04 = e;
  double* c_05 = e;
  double* c_06 = e;
  double* c_07 = e;
  double* c_08 = e;
  double* c_09 = e;
  double* c_10 = e;
  double* c_11 = e;
  double* c_12 = e;
  double* c_13 = e;
  double* c_14 = e;
  double* c_15 = e;
  double* c_16 = e;
  double* c_17 = e;
  double* c_18 = e;
  double* c_19 = e;
  double* c_20 = e;
  double* c_21 = e;
  double* c_22 = e;
  double* c_23 = e;
  double* c_24 = e;
  double* c_25 = e;
  double* c_26 = e;
  double* c_27 = e;
  double* c_28 = e;
  double* c_29 = e;
  double* c_30 = e;
  double* c_31 = e;
  double* c_32 = e;
  double* c_33 = e;
  double* c_34 = e;
  double* c_35 = e;
  double* c_36 = e;
  double* c_37 = e;
  double* c_38 = e;
  double* c_39 = e;

  RAJA::forall<RAJA::hip_exec<256>>(
      RAJA::TypedRangeSegment<int>(0, 1), [=] RAJA_DEVICE (int i) {
    double a = a_00[i];
    RAJA::atomicAdd<RAJA::hip_atomic>(c_00+i, b_00[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_01+i, b_01[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_02+i, b_02[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_03+i, b_03[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_04+i, b_04[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_05+i, b_05[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_06+i, b_06[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_07+i, b_07[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_08+i, b_08[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_09+i, b_09[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_10+i, b_10[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_11+i, b_11[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_12+i, b_12[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_13+i, b_13[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_14+i, b_14[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_15+i, b_15[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_16+i, b_16[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_17+i, b_17[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_18+i, b_18[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_19+i, b_19[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_20+i, b_20[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_21+i, b_21[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_22+i, b_22[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_23+i, b_23[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_24+i, b_24[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_25+i, b_25[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_26+i, b_26[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_27+i, b_27[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_28+i, b_28[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_29+i, b_29[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_30+i, b_30[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_31+i, b_31[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_32+i, b_32[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_33+i, b_33[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_34+i, b_34[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_35+i, b_35[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_36+i, b_36[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_37+i, b_37[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_38+i, b_38[i] + a);
    RAJA::atomicAdd<RAJA::hip_atomic>(c_39+i, b_39[i] + a);
  });

  bool success = false;
  if (d[0] == 1.0 && e[0] == 80.0) { success = true; }

  hipErrchk(hipHostFree(e));
  hipErrchk(hipHostFree(d));

  if (success) {
    std::cout << "success\n";
  } else {
    std::cout << "failure\n";
  }

  return success ? 0 : 1;
}
