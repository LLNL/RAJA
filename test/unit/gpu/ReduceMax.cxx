/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see raja/README-license.txt
 */

#include<string>
#include<iostream>
#include <cstdio>
#include <cfloat>
#include <random>

#include "RAJA/RAJA.hxx"

#define TEST_VEC_LEN  1024 * 1024

using namespace RAJA;
using namespace std;

//
// Global variables for counting tests executed/passed.
//
unsigned s_ntests_run = 0;
unsigned s_ntests_passed = 0;

int main(int argc, char *argv[])
{
   cout << "\n Begin RAJA GPU ReduceMax tests!!! " << endl;

   const int test_repeat = 10;

   //
   // Allocate and initialize managed data array
   //
   double *dvalue ;

   cudaMallocManaged((void **)&dvalue, sizeof(double)*TEST_VEC_LEN,
                     cudaMemAttachGlobal) ;
   for (int i=0; i<TEST_VEC_LEN; ++i) {
      dvalue[i] = -DBL_MAX ;
   }


   ///
   /// Define thread block size for CUDA exec policy
   ///
   const size_t block_size = 256;

////////////////////////////////////////////////////////////////////////////
// Run 3 different max reduction tests in a loop
////////////////////////////////////////////////////////////////////////////

   // current running max value
   double dcurrentMax = -DBL_MAX;

   // for setting random min values in arrays
   random_device rd;
   mt19937 mt(rd());
   uniform_real_distribution<double> dist(-10, 10);
   uniform_real_distribution<double> dist2(0, TEST_VEC_LEN-1);

   for (int tcount = 0; tcount < test_repeat; ++tcount) {

      cout << "\t tcount = " << tcount << endl;

      //
      // test 1 runs 3 reductions over a range multiple times to check
      //        that reduction value can be retrieved and then subsequent
      //        reductions can be run with the same reduction objects.
      //
      { // begin test 1

         double BIG_MAX = 500.0;
         ReduceMax< cuda_reduce<block_size>, double> dmax0(-DBL_MAX);
         ReduceMax< cuda_reduce<block_size>, double> dmax1(-DBL_MAX);
         ReduceMax< cuda_reduce<block_size>, double> dmax2(BIG_MAX);

         int loops = 16;
         for(int k=0; k < loops ; k++) {

            s_ntests_run++;

            double droll = dist(mt) ; 
            int index = int(dist2(mt));
            dvalue[index] = droll;
            dcurrentMax = RAJA_MAX(dcurrentMax, dvalue[index]); 

            forall< cuda_exec<block_size> >(0, TEST_VEC_LEN, 
               [=] __device__ (int i) {
                  dmax0.max(dvalue[i]) ;
                  dmax1.max(2*dvalue[i]) ;
                  dmax2.max(dvalue[i]) ;
            } ) ;

            if ( double(dmax0) != dcurrentMax ||
                 double(dmax1) != 2*dcurrentMax ||
                 double(dmax2) != BIG_MAX ) {
               cout << "\n TEST 1 FAILURE: tcount, k = "
                    << tcount << " , " << k << endl;
               cout << "  droll = " << droll << endl;
               cout << "\tdmax0 = " << static_cast<double>(dmax0) << " ("
                                    << dcurrentMax << ") " << endl;
               cout << "\tdmax1 = " << static_cast<double>(dmax1) << " ("
                                    << 2*dcurrentMax << ") " << endl;
               cout << "\tdmax2 = " << static_cast<double>(dmax2) << " ("
                                    << BIG_MAX << ") " << endl;
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
      { // begin test 2

         s_ntests_run++;

         RangeSegment seg0(0, TEST_VEC_LEN/2);
         RangeSegment seg1(TEST_VEC_LEN/2 + 1, TEST_VEC_LEN);

         IndexSet iset;
         iset.push_back(seg0);
         iset.push_back(seg1);

         ReduceMax< cuda_reduce<block_size>, double> dmax0(-DBL_MAX);
         ReduceMax< cuda_reduce<block_size>, double> dmax1(-DBL_MAX);

         int index = int(dist2(mt));

         double droll = dist(mt) ;
         dvalue[index] = droll;

         dcurrentMax = RAJA_MAX(dcurrentMax,dvalue[index]);

         forall< IndexSet::ExecPolicy<seq_segit, cuda_exec<block_size> > >(iset,
            [=] __device__ (int i) {
               dmax0.max(dvalue[i]) ;
               dmax1.max(2*dvalue[i]) ;
         } ) ;

         if ( double(dmax0) != dcurrentMax ||
              double(dmax1) != 2*dcurrentMax ) {
            cout << "\n TEST 2 FAILURE: tcount = " << tcount << endl;
            cout << "  droll = " << droll << endl;
            cout << "\tdmax0 = " << static_cast<double>(dmax0) << " ("
                                 << dcurrentMax << ") " << endl;
            cout << "\tdmax1 = " << static_cast<double>(dmax1) << " ("
                                 << 2*dcurrentMax << ") " << endl;
         } else {
            s_ntests_passed++;
         }

      } // end test 2

////////////////////////////////////////////////////////////////////////////

      //
      // test 3 runs 2 reductions over disjoint chunks of the array using
      //        an indexset with four range segments not aligned with
      //        warp boundaries to check that reduction mechanics don't
      //        depend on any sort of special indexing.
      //
      { // begin test 3

         s_ntests_run++;

         for (int i=0; i<TEST_VEC_LEN; ++i) {
            dvalue[i] = -DBL_MAX ;
         }
         dcurrentMax = -DBL_MAX;

         RangeSegment seg0(1, 1230);
         RangeSegment seg1(1237, 3385);
         RangeSegment seg2(4860, 10110);
         RangeSegment seg3(20490, 32003);

         IndexSet iset;
         iset.push_back(seg0);
         iset.push_back(seg1);
         iset.push_back(seg2);
         iset.push_back(seg3);

         ReduceMax< cuda_reduce<block_size>, double> dmax0(-DBL_MAX);
         ReduceMax< cuda_reduce<block_size>, double> dmax1(-DBL_MAX);

         // pick an index in one of the segments
         int index = 897;                      // seg 0
         if ( tcount % 2 == 0 ) index = 1297;  // seg 1
         if ( tcount % 3 == 0 ) index = 7853;  // seg 2
         if ( tcount % 4 == 0 ) index = 29457; // seg 3

         double droll = dist(mt) ;
         dvalue[index] = droll;

         dcurrentMax = RAJA_MAX(dcurrentMax, dvalue[index]);

         forall< IndexSet::ExecPolicy<seq_segit, cuda_exec<block_size> > >(iset,
            [=] __device__ (int i) {
               dmax0.max(dvalue[i]) ;
               dmax1.max(2*dvalue[i]) ;
         } ) ;

         if ( double(dmax0) != dcurrentMax ||
              double(dmax1) != 2*dcurrentMax ) {
            cout << "\n TEST 3 FAILURE: tcount = " << tcount << endl;
            cout << "  droll = " << droll << endl;
            cout << "\tdmax0 = " << static_cast<double>(dmax0) << " ("
                                 << dcurrentMax << ") " << endl;
            cout << "\tdmax1 = " << static_cast<double>(dmax1) << " ("
                                 << 2*dcurrentMax << ") " << endl;
         } else {
            s_ntests_passed++;
         }

      } // end test 3

   } // end test repeat loop

   ///
   /// Print total number of tests passed/run.
   ///
   cout << "\n Tests Passed / Tests Run = "
        << s_ntests_passed << " / " << s_ntests_run << endl;

   cudaFree(dvalue); 

   cout << "\n RAJA GPU ReduceMax tests DONE!!! " << endl;
   
   return 0 ;
}

