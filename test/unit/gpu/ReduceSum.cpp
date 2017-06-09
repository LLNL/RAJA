//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/README.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA GPU sum reductions.
///

#include <math.h>
#include <cfloat>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "RAJA/RAJA.hpp"

#include "Compare.hpp"

#define TEST_VEC_LEN 1024 * 1024 * 5

using namespace RAJA;
using namespace std;

//
// Global variables for counting tests executed/passed.
//
unsigned s_ntests_run = 0;
unsigned s_ntests_passed = 0;

int main(int argc, char *argv[])
{
  cout << "\n Begin RAJA GPU ReduceSum tests!!! " << endl;

  const int test_repeat = 10;

  //
  // Allocate and initialize managed data arrays
  //

  double dinit_val = 0.1;
  int iinit_val = 1;
  double *dvalue;
  double *rand_dvalue;

  int *ivalue;

  cudaMallocManaged((void **)&dvalue,
                    sizeof(double) * TEST_VEC_LEN,
                    cudaMemAttachGlobal);
  for (int i = 0; i < TEST_VEC_LEN; ++i) {
    dvalue[i] = dinit_val;
  }

  cudaMallocManaged((void **)&ivalue,
                    sizeof(int) * TEST_VEC_LEN,
                    cudaMemAttachGlobal);
  for (int i = 0; i < TEST_VEC_LEN; ++i) {
    ivalue[i] = iinit_val;
  }

  cudaMallocManaged((void **)&rand_dvalue,
                    sizeof(double) * TEST_VEC_LEN,
                    cudaMemAttachGlobal);

  ///
  /// Define thread block size for CUDA exec policy
  ///
  const size_t block_size = 256;

  ////////////////////////////////////////////////////////////////////////////
  // Run 3 different sum reduction tests in a loop
  ////////////////////////////////////////////////////////////////////////////

  for (int tcount = 0; tcount < test_repeat; ++tcount) {
    cout << "\t tcount = " << tcount << endl;

    //
    // test 1 runs 8 reductions over a range multiple times to check
    //        that reduction value can be retrieved and then subsequent
    //        reductions can be run with the same reduction objects.
    //
    {  // begin test 1

      double dtinit = 5.0;

      ReduceSum<cuda_reduce<block_size>, double> dsum0(0.0);
      ReduceSum<cuda_reduce<block_size>, double> dsum1(dtinit * 1.0);
      ReduceSum<cuda_reduce<block_size>, double> dsum2(0.0);
      ReduceSum<cuda_reduce<block_size>, double> dsum3(dtinit * 3.0);
      ReduceSum<cuda_reduce<block_size>, double> dsum4(0.0);
      ReduceSum<cuda_reduce<block_size>, double> dsum5(dtinit * 5.0);
      ReduceSum<cuda_reduce<block_size>, double> dsum6(0.0);
      ReduceSum<cuda_reduce<block_size>, double> dsum7(dtinit * 7.0);

      int loops = 2;
      for (int k = 0; k < loops; k++) {
        s_ntests_run++;

        forall<cuda_exec<block_size> >(0, TEST_VEC_LEN, [=] __device__(int i) {
          dsum0 += dvalue[i];
          dsum1 += dvalue[i] * 2.0;
          dsum2 += dvalue[i] * 3.0;
          dsum3 += dvalue[i] * 4.0;
          dsum4 += dvalue[i] * 5.0;
          dsum5 += dvalue[i] * 6.0;
          dsum6 += dvalue[i] * 7.0;
          dsum7 += dvalue[i] * 8.0;
        });

        double base_chk_val = dinit_val * double(TEST_VEC_LEN) * (k + 1);

        if (!equal(double(dsum0), base_chk_val)
            || !equal(double(dsum1), 2 * base_chk_val + (dtinit * 1.0))
            || !equal(double(dsum2), 3 * base_chk_val)
            || !equal(double(dsum3), 4 * base_chk_val + (dtinit * 3.0))
            || !equal(double(dsum4), 5 * base_chk_val)
            || !equal(double(dsum5), 6 * base_chk_val + (dtinit * 5.0))
            || !equal(double(dsum6), 7 * base_chk_val)
            || !equal(double(dsum7), 8 * base_chk_val + (dtinit * 7.0))) {
          cout << "\n TEST 1 FAILURE: tcount, k = " << tcount << " , " << k
               << endl;
          cout << setprecision(20) << "\tdsum0 = " << static_cast<double>(dsum0)
               << " (" << base_chk_val << ") " << endl;
          cout << setprecision(20) << "\tdsum1 = " << static_cast<double>(dsum1)
               << " (" << 2 * base_chk_val + (dtinit * 1.0) << ") " << endl;
          cout << setprecision(20) << "\tdsum2 = " << static_cast<double>(dsum2)
               << " (" << 3 * base_chk_val << ") " << endl;
          cout << setprecision(20) << "\tdsum3 = " << static_cast<double>(dsum3)
               << " (" << 4 * base_chk_val + (dtinit * 3.0) << ") " << endl;
          cout << setprecision(20) << "\tdsum4 = " << static_cast<double>(dsum4)
               << " (" << 5 * base_chk_val << ") " << endl;
          cout << setprecision(20) << "\tdsum5 = " << static_cast<double>(dsum5)
               << " (" << 6 * base_chk_val + (dtinit * 5.0) << ") " << endl;
          cout << setprecision(20) << "\tdsum6 = " << static_cast<double>(dsum6)
               << " (" << 7 * base_chk_val << ") " << endl;
          cout << setprecision(20) << "\tdsum7 = " << static_cast<double>(dsum7)
               << " (" << 8 * base_chk_val + (dtinit * 7.0) << ") " << endl;

        } else {
          s_ntests_passed++;
        }
      }

    }  // end test 1

    ////////////////////////////////////////////////////////////////////////////

    //
    // test 2 runs 4 reductions (2 int, 2 double) over complete array
    //        using an indexset with two range segments to check
    //        reduction object state is maintained properly across
    //        kernel invocations.
    //        Also exercises the get function call
    {  // begin test 2

      s_ntests_run++;

      RangeSegment seg0(0, TEST_VEC_LEN / 2);
      RangeSegment seg1(TEST_VEC_LEN / 2 + 1, TEST_VEC_LEN);

      IndexSet iset;
      iset.push_back(seg0);
      iset.push_back(seg1);

      double dtinit = 5.0;
      int itinit = 4;

      ReduceSum<cuda_reduce<block_size>, double> dsum0(dtinit * 1.0);
      ReduceSum<cuda_reduce<block_size>, int> isum1(itinit * 2);
      ReduceSum<cuda_reduce<block_size>, double> dsum2(dtinit * 3.0);
      ReduceSum<cuda_reduce<block_size>, int> isum3(itinit * 4);

      forallN<NestedPolicy<ExecList<IndexSet::
                                        ExecPolicy<seq_segit,
                                                   cuda_exec<block_size> > > > >(
          iset, [=] __device__(int i) {
            dsum0 += dvalue[i];
            isum1 += 2 * ivalue[i];
            dsum2 += 3 * dvalue[i];
            isum3 += 4 * ivalue[i];
          });

      double dbase_chk_val = dinit_val * double(iset.getLength());
      int ibase_chk_val = iinit_val * (iset.getLength());

      if (!equal(dsum0.get(), dbase_chk_val + (dtinit * 1.0))
          || !equal(isum1.get(), 2 * ibase_chk_val + (itinit * 2))
          || !equal(dsum2.get(), 3 * dbase_chk_val + (dtinit * 3.0))
          || !equal(isum3.get(), 4 * ibase_chk_val + (itinit * 4))) {
        cout << "\n TEST 2 FAILURE: tcount = " << tcount << endl;
        cout << setprecision(20)
             << "\tdsum0 = " << static_cast<double>(dsum0.get()) << " ("
             << dbase_chk_val + (dtinit * 1.0) << ") " << endl;
        cout << setprecision(20)
             << "\tisum1 = " << static_cast<double>(isum1.get()) << " ("
             << 2 * ibase_chk_val + (itinit * 2) << ") " << endl;
        cout << setprecision(20)
             << "\tdsum2 = " << static_cast<double>(dsum2.get()) << " ("
             << 3 * dbase_chk_val + (dtinit * 3.0) << ") " << endl;
        cout << setprecision(20)
             << "\tisum3 = " << static_cast<double>(isum3.get()) << " ("
             << 4 * ibase_chk_val + (itinit * 4) << ") " << endl;

      } else {
        s_ntests_passed++;
      }

    }  // end test 2

    ////////////////////////////////////////////////////////////////////////////

    //
    // test 3 runs 4 reductions (2 int, 2 double) over disjoint chunks
    //        of the array using an indexset with four range segments
    //        not aligned with warp boundaries to check that reduction
    //        mechanics don't depend on any sort of special indexing.
    //
    {  // begin test 3

      s_ntests_run++;

      RangeSegment seg0(1, 1230);
      RangeSegment seg1(1237, 3385);
      RangeSegment seg2(4860, 10110);
      RangeSegment seg3(20490, 32003);

      IndexSet iset;
      iset.push_back(seg0);
      iset.push_back(seg1);
      iset.push_back(seg2);
      iset.push_back(seg3);

      double dtinit = 5.0;
      int itinit = 4;

      ReduceSum<cuda_reduce<block_size>, double> dsum0(dtinit * 1.0);
      ReduceSum<cuda_reduce<block_size>, int> isum1(itinit * 2);
      ReduceSum<cuda_reduce<block_size>, double> dsum2(dtinit * 3.0);
      ReduceSum<cuda_reduce<block_size>, int> isum3(itinit * 4);

      forall<IndexSet::ExecPolicy<seq_segit, cuda_exec<block_size> > >(
          iset, [=] __device__(int i) {
            dsum0 += dvalue[i];
            isum1 += 2 * ivalue[i];
            dsum2 += 3 * dvalue[i];
            isum3 += 4 * ivalue[i];
          });

      double dbase_chk_val = dinit_val * double(iset.getLength());
      int ibase_chk_val = iinit_val * double(iset.getLength());

      if (!equal(double(dsum0), dbase_chk_val + (dtinit * 1.0))
          || !equal(int(isum1), 2 * ibase_chk_val + (itinit * 2))
          || !equal(double(dsum2), 3 * dbase_chk_val + (dtinit * 3.0))
          || !equal(int(isum3), 4 * ibase_chk_val + (itinit * 4))) {
        cout << "\n TEST 3 FAILURE: tcount = " << tcount << endl;
        cout << setprecision(20) << "\tdsum0 = " << static_cast<double>(dsum0)
             << " (" << dbase_chk_val + (dtinit * 1.0) << ") " << endl;
        cout << setprecision(20) << "\tisum1 = " << static_cast<double>(isum1)
             << " (" << 2 * ibase_chk_val + (itinit * 2) << ") " << endl;
        cout << setprecision(20) << "\tdsum2 = " << static_cast<double>(dsum2)
             << " (" << 3 * dbase_chk_val + (dtinit * 3.0) << ") " << endl;
        cout << setprecision(20) << "\tisum3 = " << static_cast<double>(isum3)
             << " (" << 4 * ibase_chk_val + (itinit * 4) << ") " << endl;

      } else {
        s_ntests_passed++;
      }

    }  // end test 3

    ////////////////////////////////////////////////////////////////////////////

    {  // Begin test4

      ReduceSum<cuda_reduce_atomic<block_size>, double> dsumN(0.0);
      ReduceSum<cuda_reduce_atomic<block_size>, double> dsumP(0.0);

      double neg_chk_val = 0.0;
      double pos_chk_val = 0.0;

      int loops = 3;
      for (int k = 0; k < loops; k++) {
        s_ntests_run++;

        for (int i = 0; i < TEST_VEC_LEN; ++i) {
          rand_dvalue[i] = drand48() - 0.5;
          if (rand_dvalue[i] < 0.0) {
            neg_chk_val += rand_dvalue[i];
          } else {
            pos_chk_val += rand_dvalue[i];
          }
        }
        forall<cuda_exec<block_size> >(0, TEST_VEC_LEN, [=] __device__(int i) {
          if (rand_dvalue[i] < 0.0) {
            dsumN += rand_dvalue[i];
          } else {
            dsumP += rand_dvalue[i];
          }
        });

        if (!equal(dsumN.get(), neg_chk_val)
            || !equal(dsumP.get(), pos_chk_val)) {
          cout << "\n TEST 4 FAILURE: tcount, k = " << tcount << " , " << k
               << endl;
          cout << setprecision(20) << "\tdsumN = " << static_cast<double>(dsumN)
               << " (" << neg_chk_val << ") " << endl;
          cout << setprecision(20) << "\tdsumP = " << static_cast<double>(dsumP)
               << " (" << pos_chk_val << ") " << endl;

        } else {
          s_ntests_passed++;
        }
      }
    }  // end test4
  }    // end test repeat loop

  ///
  /// Print total number of tests passed/run.
  ///
  cout << "\n Tests Passed / Tests Run = " << s_ntests_passed << " / "
       << s_ntests_run << endl;

  cudaFree(dvalue);
  cudaFree(ivalue);

  return !(s_ntests_passed == s_ntests_run);
}
