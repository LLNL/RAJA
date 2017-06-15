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
/// Source file containing tests for RAJA GPU min-loc reductions.
///

#include <cfloat>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>

#include "RAJA/RAJA.hpp"

#define TEST_VEC_LEN 1024 * 1024 * 7

typedef struct {
  double val;
  int idx;
} minloc_t;

using namespace RAJA;
using namespace std;

//
// Global variables for counting tests executed/passed.
//
unsigned s_ntests_run = 0;
unsigned s_ntests_passed = 0;

int main(int argc, char *argv[])
{
  cout << "\n Begin RAJA GPU ReduceMinLoc tests!!! " << endl;

  const int test_repeat = 10;

  //
  // Allocate and initialize managed data array
  //
  double *dvalue;

  cudaMallocManaged((void **)&dvalue,
                    sizeof(double) * TEST_VEC_LEN,
                    cudaMemAttachGlobal);
  for (int i = 0; i < TEST_VEC_LEN; ++i) {
    dvalue[i] = DBL_MAX;
  }

  ///
  /// Define thread block size for CUDA exec policy
  ///
  const size_t block_size = 256;

  ////////////////////////////////////////////////////////////////////////////
  // Run 3 different min reduction tests in a loop
  ////////////////////////////////////////////////////////////////////////////

  // current running min value

  minloc_t dcurrentMin;
  dcurrentMin.val = DBL_MAX;
  dcurrentMin.idx = -1;

  // for setting random min values in arrays
  random_device rd;
  mt19937 mt(rd());
  uniform_real_distribution<double> dist(-10, 10);
  uniform_real_distribution<double> dist2(0, TEST_VEC_LEN - 1);

  for (int tcount = 0; tcount < test_repeat; ++tcount) {
    cout << "\t tcount = " << tcount << endl;

    //
    // test 1 runs 3 reductions over a range multiple times to check
    //        that reduction value can be retrieved and then subsequent
    //        reductions can be run with the same reduction objects.
    //        Also exercises the get function call
    {  // begin test 1

      double BIG_MIN = -500.0;
      ReduceMinLoc<cuda_reduce<block_size>, double> dmin0(DBL_MAX, -1);
      ReduceMinLoc<cuda_reduce<block_size>, double> dmin1(DBL_MAX, 1);
      ReduceMinLoc<cuda_reduce<block_size>, double> dmin2(BIG_MIN, -1);

      int loops = 16;
      for (int k = 0; k < loops; k++) {
        s_ntests_run++;

        double droll = dist(mt);
        int index = int(dist2(mt));
        dvalue[index] = droll;

        minloc_t lmin = {droll, index};
        dvalue[index] = droll;
        dcurrentMin = RAJA_MINLOC(dcurrentMin, lmin);
        // printf("droll %lf : dcurrentMin %lf : index
        // %d\n",droll,dcurrentMin,index);
        forall<cuda_exec<block_size> >(0, TEST_VEC_LEN, [=] __device__(int i) {
          dmin0.minloc(dvalue[i], i);
          dmin1.minloc(2 * dvalue[i], i);
          dmin2.minloc(dvalue[i], i);
        });

        if (dmin0.get() != dcurrentMin.val || dmin1.get() != 2 * dcurrentMin.val
            || dmin2.get() != BIG_MIN
            || dmin0.getLoc() != dcurrentMin.idx
            || dmin1.getLoc() != dcurrentMin.idx) {
          cout << "\n TEST 1 FAILURE: tcount, k = " << tcount << " , " << k
               << endl;
          cout << "  droll = " << droll << endl;
          cout << "\tdmin0 = " << static_cast<double>(dmin0.get()) << " ("
               << dcurrentMin.val << ") " << endl;
          cout << "\tdmin1 = " << static_cast<double>(dmin1.get()) << " ("
               << 2 * dcurrentMin.val << ") " << endl;
          cout << "\tdmin2 = " << static_cast<double>(dmin2.get()) << " ("
               << BIG_MIN << ") " << endl;
        } else {
          s_ntests_passed++;
        }
      }

    }  // end test 1
    ////////////////////////////////////////////////////////////////////////////

    //
    // test 2 runs 2 reductions over complete array using an indexset
    //        with two range segments to check reduction object state
    //        is maintained properly across kernel invocations.
    //
    {  // begin test 2

      s_ntests_run++;

      RangeSegment seg0(0, TEST_VEC_LEN / 2);
      RangeSegment seg1(TEST_VEC_LEN / 2 + 1, TEST_VEC_LEN);

      IndexSet iset;
      iset.push_back(seg0);
      iset.push_back(seg1);

      ReduceMinLoc<cuda_reduce<block_size>, double> dmin0(DBL_MAX, -1);
      ReduceMinLoc<cuda_reduce<block_size>, double> dmin1(DBL_MAX, -1);

      int index = int(dist2(mt));

      double droll = dist(mt);
      dvalue[index] = droll;

      minloc_t lmin = {droll, index};
      dvalue[index] = droll;
      dcurrentMin = RAJA_MINLOC(dcurrentMin, lmin);

      forall<IndexSet::ExecPolicy<seq_segit, cuda_exec<block_size> > >(
          iset, [=] __device__(int i) {
            dmin0.minloc(dvalue[i], i);
            dmin1.minloc(2 * dvalue[i], i);
          });

      if (double(dmin0) != dcurrentMin.val
          || double(dmin1) != 2 * dcurrentMin.val
          || dmin0.getLoc() != dcurrentMin.idx
          || dmin1.getLoc() != dcurrentMin.idx) {
        cout << "\n TEST 2 FAILURE: tcount = " << tcount << endl;
        cout << "   droll = " << droll << endl;
        cout << "\tdmin0 = " << static_cast<double>(dmin0) << " ("
             << dcurrentMin.val << ") " << endl;
        cout << "\tdmin1 = " << static_cast<double>(dmin1) << " ("
             << 2 * dcurrentMin.val << ") " << endl;
      } else {
        s_ntests_passed++;
      }

    }  // end test 2

    ///////////////////////////////////////////////////////////////////////

    //
    // test 3 runs 2 reductions over disjoint chunks of the array using
    //        an indexset with four range segments not aligned with
    //        warp boundaries to check that reduction mechanics don't
    //        depend on any sort of special indexing.
    //
    {  // begin test 3

      s_ntests_run++;

      for (int i = 0; i < TEST_VEC_LEN; ++i) {
        dvalue[i] = DBL_MAX;
      }

      dcurrentMin.val = DBL_MAX;
      dcurrentMin.idx = -1;

      RangeSegment seg0(1, 1230);
      RangeSegment seg1(1237, 3385);
      RangeSegment seg2(4860, 10110);
      RangeSegment seg3(20490, 32003);

      IndexSet iset;
      iset.push_back(seg0);
      iset.push_back(seg1);
      iset.push_back(seg2);
      iset.push_back(seg3);

      ReduceMinLoc<cuda_reduce<block_size>, double> dmin0(DBL_MAX, -1);
      ReduceMinLoc<cuda_reduce<block_size>, double> dmin1(DBL_MAX, -1);

      // pick an index in one of the segments
      int index = 897;                     // seg 0
      if (tcount % 2 == 0) index = 1297;   // seg 1
      if (tcount % 3 == 0) index = 7853;   // seg 2
      if (tcount % 4 == 0) index = 29457;  // seg 3

      double droll = dist(mt);
      dvalue[index] = droll;

      minloc_t lmin = {droll, index};
      dvalue[index] = droll;
      dcurrentMin = RAJA_MINLOC(dcurrentMin, lmin);

      forall<IndexSet::ExecPolicy<seq_segit, cuda_exec<block_size> > >(
          iset, [=] __device__(int i) {
            dmin0.minloc(dvalue[i], i);
            dmin1.minloc(2 * dvalue[i], i);
          });

      if (double(dmin0) != dcurrentMin.val
          || double(dmin1) != 2 * dcurrentMin.val
          || dmin0.getLoc() != dcurrentMin.idx
          || dmin1.getLoc() != dcurrentMin.idx) {
        cout << "\n TEST 3 FAILURE: tcount = " << tcount << endl;
        cout << "   droll = " << droll << endl;
        cout << "\tdmin0 = " << static_cast<double>(dmin0) << " ("
             << dcurrentMin.val << ") " << endl;
        cout << "\tdmin1 = " << static_cast<double>(dmin1) << " ("
             << 2 * dcurrentMin.val << ") " << endl;
      } else {
        s_ntests_passed++;
      }

    }  // end test 3
  }    // end test repeat loop

  ///
  /// Print total number of tests passed/run.
  ///
  cout << "\n Tests Passed / Tests Run = " << s_ntests_passed << " / "
       << s_ntests_run << endl;

  cudaFree(dvalue);

  cout << "\n RAJA GPU ReduceMinLoc tests DONE!!! " << endl;

  return !(s_ntests_passed == s_ntests_run);
}
