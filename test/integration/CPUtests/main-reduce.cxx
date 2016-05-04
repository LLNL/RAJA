/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see raja/README-license.txt
 */

//
// Main program illustrating simple RAJA index set creation 
// and execution and methods.
//

#include <cstdlib>
#include <time.h>
#include <cmath>

#include<string>
#include<vector>
#include<iostream>

#include "RAJA/RAJA.hxx"

using namespace RAJA;
using namespace std;

#include "Compare.hxx"

#include "buildIndexSet.hxx"


//
// Global variables for counting tests executed/passed.
//
unsigned s_ntests_run_total = 0;
unsigned s_ntests_passed_total = 0;

unsigned s_ntests_run = 0;
unsigned s_ntests_passed = 0;


//=========================================================================
//=========================================================================
// 
// Methods that define and run various RAJA reduction tests
// 
//=========================================================================
//=========================================================================

///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA min reduction tests 
// based on execution policy template
//
///////////////////////////////////////////////////////////////////////////
template <typename ISET_POLICY_T,
          typename REDUCE_POLICY_T>
void runBasicMinReductionTest(const string& policy,
                              Real_ptr in_array, Index_type alen,
                              const IndexSet& iset,
                              const RAJAVec<Index_type>& is_indices)
{
   Real_ptr test_array;
   posix_memalign((void **)&test_array, DATA_ALIGN, alen*sizeof(Real_type)) ;

   //
   // Make all test array values positve
   //
   for (Index_type i=0 ; i<alen; ++i) {
      test_array[i] = fabs( in_array[i] );
   }

   //
   // Generate reference result for min in middle of index set.
   //
   const Index_type ref_min_indx = Index_type(is_indices[is_indices.size()/2]);
   const Real_type  ref_min_val  = -100.0;

   test_array[ref_min_indx] = ref_min_val;

#if 0
   cout << "ref_min_indx = " << ref_min_indx << endl;
   cout << "ref_min_val = " << ref_min_val << endl;
   cout << "test_array[ref_min_indx] = " 
             << test_array[ref_min_indx] << endl;
#endif 

   cout << "\n Test MIN reduction for " << policy << "\n";

   ReduceMin<REDUCE_POLICY_T, Real_type> tmin0(1.0e+20);
   ReduceMin<REDUCE_POLICY_T, Real_type> tmin1(-200.0);

   int loops = 2;

   for (int k = 1; k <= loops; ++ k) {

      s_ntests_run++;
      s_ntests_run_total++;

//    cout << "k = " << k << endl;

      forall< ISET_POLICY_T >( iset, [=] (Index_type idx) {
         tmin0.min(k*test_array[idx]);
         tmin1.min(test_array[idx]);
      } );

      if ( Real_type(tmin0) != Real_type(k*ref_min_val) ||
           Real_type(tmin1) != Real_type(-200.0) ) {
         cout << "\n TEST FAILURE: k = " << k << endl;
         cout << "\ttmin0 = " << Real_type(tmin0) << " ("
                              << k*ref_min_val << ") " << endl;
         cout << "\ttmin1 = " << Real_type(tmin1) << " ("
                              << -200.0 << ") " << endl;
      } else {
         s_ntests_passed++;
         s_ntests_passed_total++;
      }

   }

   free(test_array); 
}

 
///////////////////////////////////////////////////////////////////////////
//
// Run RAJA min reduction tests with available RAJA execution policies....
//
///////////////////////////////////////////////////////////////////////////
void runMinReduceTests( Real_ptr in_array,
                        Index_type alen,
                        const IndexSet& iset,
                        const RAJAVec<Index_type>& is_indices )
{
   cout << "\n\n   BEGIN RAJA::forall MIN REDUCE tests...." << endl;

   // initialize test counters for this test set
   s_ntests_run = 0; 
   s_ntests_passed = 0; 

   runBasicMinReductionTest< 
      IndexSet::ExecPolicy<seq_segit, seq_exec>, seq_reduce > ( 
               "ExecPolicy<seq_segit, seq_exec>",
                in_array, alen,
                iset, is_indices ); 

   runBasicMinReductionTest< 
      IndexSet::ExecPolicy<seq_segit, simd_exec>, seq_reduce > ( 
               "ExecPolicy<seq_segit, simd_exec>",
                in_array, alen,
                iset, is_indices ); 

#ifdef RAJA_USE_OPENMP
   runBasicMinReductionTest< 
      IndexSet::ExecPolicy<seq_segit, omp_parallel_for_exec>, omp_reduce > ( 
               "ExecPolicy<seq_segit, omp_parallel_for_exec>",
                in_array, alen,
                iset, is_indices ); 

   runBasicMinReductionTest< 
      IndexSet::ExecPolicy<omp_parallel_for_segit, seq_exec>, omp_reduce > ( 
               "ExecPolicy<omp_parallel_for_segit, seq_exec>",
                in_array, alen,
                iset, is_indices ); 

   runBasicMinReductionTest< 
      IndexSet::ExecPolicy<omp_parallel_for_segit, simd_exec>, omp_reduce > ( 
               "ExecPolicy<omp_parallel_for_segit, simd_exec>",
                in_array, alen,
                iset, is_indices ); 
#endif

#ifdef RAJA_USE_CILK
   runBasicMinReductionTest<
      IndexSet::ExecPolicy<seq_segit, cilk_for_exec>, cilk_reduce > (
               "ExecPolicy<seq_segit, cilk_for_exec>",
                in_array, alen,
                iset, is_indices );

   runBasicMinReductionTest<
      IndexSet::ExecPolicy<cilk_for_segit, seq_exec>, cilk_reduce > (
               "ExecPolicy<cilk_for_segit, seq_exec>",
                in_array, alen,
                iset, is_indices );

   runBasicMinReductionTest<
      IndexSet::ExecPolicy<cilk_for_segit, simd_exec>, cilk_reduce > (
               "ExecPolicy<cilk_for_segit, simd_exec>",
                in_array, alen,
                iset, is_indices );
#endif

   cout << "\n tests passed / test run: " 
             << s_ntests_passed << " / " << s_ntests_run << endl; 

   cout << "\n   END RAJA::forall MIN REDUCE tests... " << endl;
}


///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA min-loc reduction tests 
// based on execution policy template
//
///////////////////////////////////////////////////////////////////////////
template <typename ISET_POLICY_T,
          typename REDUCE_POLICY_T>
void runBasicMinLocReductionTest(const string& policy,
                                 Real_ptr in_array, Index_type alen,
                                 const IndexSet& iset,
                                 const RAJAVec<Index_type>& is_indices)
{
   Real_ptr test_array;
   posix_memalign((void **)&test_array, DATA_ALIGN, alen*sizeof(Real_type)) ;

   //
   // Make all test array values positve
   //
   for (Index_type i=0 ; i<alen; ++i) {
      test_array[i] = fabs( in_array[i] );
   }

   //
   // Generate reference result for min in middle of index set.
   //
   const Index_type ref_min_indx = Index_type(is_indices[is_indices.size()/2]);
   const Real_type  ref_min_val  = -100.0;

   test_array[ref_min_indx] = ref_min_val;

#if 0
   cout << "ref_min_indx = " << ref_min_indx << endl;
   cout << "ref_min_val = " << ref_min_val << endl;
   cout << "test_array[ref_min_indx] = " 
             << test_array[ref_min_indx] << endl;
#endif 

   cout << "\n Test MIN-LOC reduction for " << policy << "\n";

   ReduceMinLoc<REDUCE_POLICY_T, Real_type> tmin0(1.0e+20, -1);
   ReduceMinLoc<REDUCE_POLICY_T, Real_type> tmin1(-200.0, -1);

   int loops = 2;

   for (int k = 1; k <= loops; ++ k) {

      s_ntests_run++;
      s_ntests_run_total++;

//    cout << "k = " << k << endl;

      forall< ISET_POLICY_T >( iset, [=] (Index_type idx) {
         tmin0.minloc(k*test_array[idx], idx);
         tmin1.minloc(test_array[idx], idx);
      } );

      if ( Real_type(tmin0) != Real_type(k*ref_min_val) ||
           tmin0.getMinLoc() != ref_min_indx ||
           Real_type(tmin1) != Real_type(-200.0) || 
           tmin1.getMinLoc() != -1 ) {
         cout << "\n TEST FAILURE: k = " << k << endl;
         cout << "\ttmin0, loc = " 
              << Real_type(tmin0) << " , " << tmin0.getMinLoc()
              << " (" << k*ref_min_val << ", "
                      << ref_min_indx << " ) " << endl;
         cout << "\ttmin1, loc = " 
              << Real_type(tmin1) << " , " << tmin1.getMinLoc()
              << " (" << -200.0 << ", " << -1 << " ) " << endl;
      } else {
         s_ntests_passed++;
         s_ntests_passed_total++;
      }

   }

   free(test_array); 
}


///////////////////////////////////////////////////////////////////////////
//
// Run RAJA min-loc reduction tests with available RAJA execution policies....
//
///////////////////////////////////////////////////////////////////////////
void runMinLocReduceTests( Real_ptr in_array,
                           Index_type alen,
                           const IndexSet& iset,
                           const RAJAVec<Index_type>& is_indices )
{
   cout << "\n\n   BEGIN RAJA::forall MIN-LOC REDUCE tests...." << endl;

   // initialize test counters for this test set
   s_ntests_run = 0; 
   s_ntests_passed = 0; 

   runBasicMinLocReductionTest< 
      IndexSet::ExecPolicy<seq_segit, seq_exec>, seq_reduce > ( 
               "ExecPolicy<seq_segit, seq_exec>",
                in_array, alen,
                iset, is_indices ); 

   runBasicMinLocReductionTest< 
      IndexSet::ExecPolicy<seq_segit, simd_exec>, seq_reduce > ( 
               "ExecPolicy<seq_segit, simd_exec>",
                in_array, alen,
                iset, is_indices ); 

#ifdef RAJA_USE_OPENMP
   runBasicMinLocReductionTest< 
      IndexSet::ExecPolicy<seq_segit, omp_parallel_for_exec>, omp_reduce > ( 
               "ExecPolicy<seq_segit, omp_parallel_for_exec>",
                in_array, alen,
                iset, is_indices ); 

   runBasicMinLocReductionTest< 
      IndexSet::ExecPolicy<omp_parallel_for_segit, seq_exec>, omp_reduce > ( 
               "ExecPolicy<omp_parallel_for_segit, seq_exec>",
                in_array, alen,
                iset, is_indices ); 

   runBasicMinLocReductionTest< 
      IndexSet::ExecPolicy<omp_parallel_for_segit, simd_exec>, omp_reduce > ( 
               "ExecPolicy<omp_parallel_for_segit, simd_exec>",
                in_array, alen,
                iset, is_indices ); 
#endif

#ifdef RAJA_USE_CILK
   runBasicMinLocReductionTest<
      IndexSet::ExecPolicy<seq_segit, cilk_for_exec>, cilk_reduce > (
               "ExecPolicy<seq_segit, cilk_for_exec>",
                in_array, alen,
                iset, is_indices );

   runBasicMinLocReductionTest<
      IndexSet::ExecPolicy<cilk_for_segit, seq_exec>, cilk_reduce > (
               "ExecPolicy<cilk_for_segit, seq_exec>",
                in_array, alen,
                iset, is_indices );

   runBasicMinLocReductionTest<
      IndexSet::ExecPolicy<cilk_for_segit, simd_exec>, cilk_reduce > (
               "ExecPolicy<cilk_for_segit, simd_exec>",
                in_array, alen,
                iset, is_indices );
#endif

   cout << "\n tests passed / test run: " 
             << s_ntests_passed << " / " << s_ntests_run << endl; 

   cout << "\n   END RAJA::forall MIN-LOC REDUCE tests... " << endl;
}


///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA max reduction tests 
// based on execution policy template
//
///////////////////////////////////////////////////////////////////////////
template <typename ISET_POLICY_T,
          typename REDUCE_POLICY_T>
void runBasicMaxReductionTest(const string& policy,
                              Real_ptr in_array, Index_type alen,
                              const IndexSet& iset,
                              const RAJAVec<Index_type>& is_indices)
{
   Real_ptr test_array;
   posix_memalign((void **)&test_array, DATA_ALIGN, alen*sizeof(Real_type)) ;

   //
   // Make all test array values negative
   //
   for (Index_type i=0 ; i<alen; ++i) {
      test_array[i] = -fabs( in_array[i] );
   }

   //
   // Generate reference result for max in middle of index set.
   //
   const Index_type ref_max_indx = Index_type(is_indices[is_indices.size()/2]);
   const Real_type  ref_max_val  = 100.0;

   test_array[ref_max_indx] = ref_max_val;

#if 0
   cout << "ref_max_indx = " << ref_max_indx << endl;
   cout << "ref_max_val = " << ref_max_val << endl;
   cout << "test_array[ref_max_indx] = " 
             << test_array[ref_max_indx] << endl;
#endif 

   cout << "\n Test MAX reduction for " << policy << "\n";

   ReduceMax<REDUCE_POLICY_T, Real_type> tmax0(-1.0e+20);
   ReduceMax<REDUCE_POLICY_T, Real_type> tmax1(200.0);

   int loops = 2;

   for (int k = 1; k <= loops; ++ k) {

      s_ntests_run++;
      s_ntests_run_total++;

//    cout << "k = " << k << endl;

      forall< ISET_POLICY_T >( iset, [=] (Index_type idx) {
         tmax0.max(k*test_array[idx]);
         tmax1.max(test_array[idx]);
      } );

      if ( Real_type(tmax0) != Real_type(k*ref_max_val) ||
           Real_type(tmax1) != Real_type(200.0) ) {
         cout << "\n TEST FAILURE: k = " << k << endl;
         cout << "\ttmax0 = " << Real_type(tmax0) << " ("
                              << k*ref_max_val << ") " << endl;
         cout << "\ttmax1 = " << Real_type(tmax1) << " ("
                              << 200.0 << ") " << endl;
      } else {
         s_ntests_passed++;
         s_ntests_passed_total++;
      }

   }

   free(test_array); 
}

 
///////////////////////////////////////////////////////////////////////////
//
// Run RAJA max reduce tests with available RAJA execution policies....
//
///////////////////////////////////////////////////////////////////////////
void runMaxReduceTests( Real_ptr in_array,
                        Index_type alen,
                        const IndexSet& iset,
                        const RAJAVec<Index_type>& is_indices )
{
   cout << "\n\n   BEGIN RAJA::forall MAX REDUCE tests...." << endl;

   // initialize test counters for this test set
   s_ntests_run = 0; 
   s_ntests_passed = 0; 

   runBasicMaxReductionTest< 
      IndexSet::ExecPolicy<seq_segit, seq_exec>, seq_reduce > ( 
               "ExecPolicy<seq_segit, seq_exec>",
                in_array, alen,
                iset, is_indices ); 

   runBasicMaxReductionTest< 
      IndexSet::ExecPolicy<seq_segit, simd_exec>, seq_reduce > ( 
               "ExecPolicy<seq_segit, simd_exec>",
                in_array, alen,
                iset, is_indices ); 

#ifdef RAJA_USE_OPENMP
   runBasicMaxReductionTest< 
      IndexSet::ExecPolicy<seq_segit, omp_parallel_for_exec>, omp_reduce > ( 
               "ExecPolicy<seq_segit, omp_parallel_for_exec>",
                in_array, alen,
                iset, is_indices ); 

   runBasicMaxReductionTest< 
      IndexSet::ExecPolicy<omp_parallel_for_segit, seq_exec>, omp_reduce > ( 
               "ExecPolicy<omp_parallel_for_segit, seq_exec>",
                in_array, alen,
                iset, is_indices ); 

   runBasicMaxReductionTest< 
      IndexSet::ExecPolicy<omp_parallel_for_segit, simd_exec>, omp_reduce > ( 
               "ExecPolicy<omp_parallel_for_segit, simd_exec>",
                in_array, alen,
                iset, is_indices ); 
#endif

#ifdef RAJA_USE_CILK
   runBasicMaxReductionTest<
      IndexSet::ExecPolicy<seq_segit, cilk_for_exec>, cilk_reduce > (
               "ExecPolicy<seq_segit, cilk_for_exec>",
                in_array, alen,
                iset, is_indices );

   runBasicMaxReductionTest<
      IndexSet::ExecPolicy<cilk_for_segit, seq_exec>, cilk_reduce > (
               "ExecPolicy<cilk_for_segit, seq_exec>",
                in_array, alen,
                iset, is_indices );

   runBasicMaxReductionTest<
      IndexSet::ExecPolicy<cilk_for_segit, simd_exec>, cilk_reduce > (
               "ExecPolicy<cilk_for_segit, simd_exec>",
                in_array, alen,
                iset, is_indices );
#endif

   cout << "\n tests passed / test run: " 
             << s_ntests_passed << " / " << s_ntests_run << endl; 
   

   cout << "\n   END RAJA::forall MAX REDUCE tests... " << endl;
}


///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA max-loc reduction tests 
// based on execution policy template
//
///////////////////////////////////////////////////////////////////////////
template <typename ISET_POLICY_T,
          typename REDUCE_POLICY_T>
void runBasicMaxLocReductionTest(const string& policy,
                                 Real_ptr in_array, Index_type alen,
                                 const IndexSet& iset,
                                 const RAJAVec<Index_type>& is_indices)
{
   Real_ptr test_array;
   posix_memalign((void **)&test_array, DATA_ALIGN, alen*sizeof(Real_type)) ;

   //
   // Make all test array values negative
   //
   for (Index_type i=0 ; i<alen; ++i) {
      test_array[i] = -fabs( in_array[i] );
   }

   //
   // Generate reference result for min in middle of index set.
   //
   const Index_type ref_max_indx = Index_type(is_indices[is_indices.size()/2]);
   const Real_type  ref_max_val  = 100.0;

   test_array[ref_max_indx] = ref_max_val;

#if 0
   cout << "ref_max_indx = " << ref_max_indx << endl;
   cout << "ref_max_val = " << ref_max_val << endl;
   cout << "test_array[ref_max_indx] = " 
             << test_array[ref_max_indx] << endl;
#endif 

   cout << "\n Test MAX-LOC reduction for " << policy << "\n";

   ReduceMaxLoc<REDUCE_POLICY_T, Real_type> tmax0(-1.0e+20, -1);
   ReduceMaxLoc<REDUCE_POLICY_T, Real_type> tmax1(200.0, -1);

   int loops = 2;

   for (int k = 1; k <= loops; ++ k) {

      s_ntests_run++;
      s_ntests_run_total++;

//    cout << "k = " << k << endl;

      forall< ISET_POLICY_T >( iset, [=] (Index_type idx) {
         tmax0.maxloc(k*test_array[idx], idx);
         tmax1.maxloc(test_array[idx], idx);
      } );

      if ( Real_type(tmax0) != Real_type(k*ref_max_val) ||
           tmax0.getMaxLoc() != ref_max_indx ||
           Real_type(tmax1) != Real_type(200.0) || 
           tmax1.getMaxLoc() != -1 ) {
         cout << "\n TEST FAILURE: k = " << k << endl;
         cout << "\ttmax0, loc = " 
              << Real_type(tmax0) << " , " << tmax0.getMaxLoc()
              << " (" << k*ref_max_val << ", "
                      << ref_max_indx << " ) " << endl;
         cout << "\ttmax1, loc = "
              << Real_type(tmax1) << " , " << tmax1.getMaxLoc()
              << " (" << 200.0 << ", " << -1 << " ) " << endl;
      } else {
         s_ntests_passed++;
         s_ntests_passed_total++;
      }

   }

   free(test_array); 
}


///////////////////////////////////////////////////////////////////////////
//
// Run RAJA max-loc reduction tests with available RAJA execution policies....
//
///////////////////////////////////////////////////////////////////////////
void runMaxLocReduceTests( Real_ptr in_array,
                           Index_type alen,
                           const IndexSet& iset,
                           const RAJAVec<Index_type>& is_indices )
{
   cout << "\n\n   BEGIN RAJA::forall MAX-LOC REDUCE tests...." << endl;

   // initialize test counters for this test set
   s_ntests_run = 0; 
   s_ntests_passed = 0; 

   runBasicMaxLocReductionTest< 
      IndexSet::ExecPolicy<seq_segit, seq_exec>, seq_reduce > ( 
               "ExecPolicy<seq_segit, seq_exec>",
                in_array, alen,
                iset, is_indices ); 

   runBasicMaxLocReductionTest< 
      IndexSet::ExecPolicy<seq_segit, simd_exec>, seq_reduce > ( 
               "ExecPolicy<seq_segit, simd_exec>",
                in_array, alen,
                iset, is_indices ); 

#ifdef RAJA_USE_OPENMP
   runBasicMaxLocReductionTest< 
      IndexSet::ExecPolicy<seq_segit, omp_parallel_for_exec>, omp_reduce > ( 
               "ExecPolicy<seq_segit, omp_parallel_for_exec>",
                in_array, alen,
                iset, is_indices ); 

   runBasicMaxLocReductionTest< 
      IndexSet::ExecPolicy<omp_parallel_for_segit, seq_exec>, omp_reduce > ( 
               "ExecPolicy<omp_parallel_for_segit, seq_exec>",
                in_array, alen,
                iset, is_indices ); 

   runBasicMaxLocReductionTest< 
      IndexSet::ExecPolicy<omp_parallel_for_segit, simd_exec>, omp_reduce > ( 
               "ExecPolicy<omp_parallel_for_segit, simd_exec>",
                in_array, alen,
                iset, is_indices ); 
#endif

#ifdef RAJA_USE_CILK
   runBasicMaxLocReductionTest<
      IndexSet::ExecPolicy<seq_segit, cilk_for_exec>, cilk_reduce > (
               "ExecPolicy<seq_segit, cilk_for_exec>",
                in_array, alen,
                iset, is_indices );

   runBasicMaxLocReductionTest<
      IndexSet::ExecPolicy<cilk_for_segit, seq_exec>, cilk_reduce > (
               "ExecPolicy<cilk_for_segit, seq_exec>",
                in_array, alen,
                iset, is_indices );

   runBasicMaxLocReductionTest<
      IndexSet::ExecPolicy<cilk_for_segit, simd_exec>, cilk_reduce > (
               "ExecPolicy<cilk_for_segit, simd_exec>",
                in_array, alen,
                iset, is_indices );
#endif

   cout << "\n tests passed / test run: " 
             << s_ntests_passed << " / " << s_ntests_run << endl; 

   cout << "\n   END RAJA::forall MAX-LOC REDUCE tests... " << endl;
}


///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA sum reduction tests 
// based on execution policy template
//
///////////////////////////////////////////////////////////////////////////
template <typename ISET_POLICY_T,
          typename REDUCE_POLICY_T>
void runBasicSumReductionTest(const string& policy,
                              Real_ptr in_array, Index_type alen,
                              const IndexSet& iset,
                              const RAJAVec<Index_type>& is_indices)
{
   //
   // Generate reference result for sum
   //
   Real_type  ref_sum  = 0.0;

   for (Index_type i = 0; i < is_indices.size(); ++i) {
      ref_sum += in_array[ is_indices[i] ];
   }

#if 0
   cout << "ref_sum = " << ref_sum << endl;
#endif 

   cout << "\n Test SUM reduction for " << policy << "\n";

   ReduceSum<REDUCE_POLICY_T, Real_type> tsum0(0.0);
   ReduceSum<REDUCE_POLICY_T, Real_type> tsum1(5.0);

   int loops = 2;

   for (int k = 1; k <= loops; ++ k) {

      s_ntests_run++;
      s_ntests_run_total++;

//    cout << "k = " << k << endl;

      forall< ISET_POLICY_T >( iset, [=] (Index_type idx) {
         tsum0 += in_array[idx];
         tsum1 += 1.0;
      } );

      if ( !equal( Real_type(tsum0), Real_type(k*ref_sum) ) ||
           !equal( Real_type(tsum1), Real_type(k*iset.getLength() + 5.0) ) ) {
         cout << "\n TEST FAILURE: k = " << k << endl;
         cout << "\ttmin0 = " << Real_type(tsum0) << " ("
                              << k*ref_sum << ") " << endl;
         cout << "\ttmin1 = " << Real_type(tsum1) << " ("
                              << k*iset.getLength() + 5.0 << ") " << endl;
      } else {
         s_ntests_passed++;
         s_ntests_passed_total++;
      }

   }
}


///////////////////////////////////////////////////////////////////////////
//
// Run RAJA sum reduce tests with available RAJA execution policies....
//
///////////////////////////////////////////////////////////////////////////
void runSumReduceTests( Real_ptr in_array,
                        Index_type alen,
                        const IndexSet& iset,
                        const RAJAVec<Index_type>& is_indices )
{
   cout << "\n\n   BEGIN RAJA::forall SUM REDUCE tests...." << endl;

   // initialize test counters for this test set
   s_ntests_run = 0; 
   s_ntests_passed = 0; 

   runBasicSumReductionTest< 
      IndexSet::ExecPolicy<seq_segit, seq_exec>, seq_reduce > ( 
               "ExecPolicy<seq_segit, seq_exec>",
                in_array, alen,
                iset, is_indices ); 

   runBasicSumReductionTest< 
      IndexSet::ExecPolicy<seq_segit, simd_exec>, seq_reduce > ( 
               "ExecPolicy<seq_segit, simd_exec>",
                in_array, alen,
                iset, is_indices ); 

#ifdef RAJA_USE_OPENMP
   runBasicSumReductionTest< 
      IndexSet::ExecPolicy<seq_segit, omp_parallel_for_exec>, omp_reduce > ( 
               "ExecPolicy<seq_segit, omp_parallel_for_exec>",
                in_array, alen,
                iset, is_indices ); 

   runBasicSumReductionTest< 
      IndexSet::ExecPolicy<omp_parallel_for_segit, seq_exec>, omp_reduce > ( 
               "ExecPolicy<omp_parallel_for_segit, seq_exec>",
                in_array, alen,
                iset, is_indices ); 

   runBasicSumReductionTest< 
      IndexSet::ExecPolicy<omp_parallel_for_segit, simd_exec>, omp_reduce > ( 
               "ExecPolicy<omp_parallel_for_segit, simd_exec>",
                in_array, alen,
                iset, is_indices ); 
#endif

#ifdef RAJA_USE_CILK
   runBasicSumReductionTest<
      IndexSet::ExecPolicy<seq_segit, cilk_for_exec>, cilk_reduce > (
               "ExecPolicy<seq_segit, cilk_for_exec>",
                in_array, alen,
                iset, is_indices );

   runBasicSumReductionTest<
      IndexSet::ExecPolicy<cilk_for_segit, seq_exec>, cilk_reduce > (
               "ExecPolicy<cilk_for_segit, seq_exec>",
                in_array, alen,
                iset, is_indices );

   runBasicSumReductionTest<
      IndexSet::ExecPolicy<cilk_for_segit, simd_exec>, cilk_reduce > (
               "ExecPolicy<cilk_for_segit, simd_exec>",
                in_array, alen,
                iset, is_indices );
#endif

   cout << "\n tests passed / test run: " 
             << s_ntests_passed << " / " << s_ntests_run << endl; 
   

   cout << "\n   END RAJA::forall SUM REDUCE tests... " << endl;
}


///////////////////////////////////////////////////////////////////////////
//
// Main Program.
//
///////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{

   //
   // Record maximum index in IndexSets for proper array allocation later.
   //
   Index_type last_indx = 0;

//
//  All methods to construct index sets should generate equivalent results.
//
   IndexSet index[NumBuildMethods];
   for (unsigned ibuild = 0; ibuild < NumBuildMethods; ++ibuild) {
      last_indx = max( last_indx,
         buildIndexSet( index, static_cast<IndexSetBuildMethod>(ibuild) ) );
#if 0 // print index set for debugging
      cout << "\n\nIndexSet( " << ibuild << " ) " << endl;
      index[ibuild].print(cout);
#endif
   }

///////////////////////////////////////////////////////////////////////////
//
// Checks for equality of all constructed index sets are performed
// in traversal test program.
//
///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////
//
// Run RAJA::forall reduction tests...
//
///////////////////////////////////////////////////////////////////////////

   IndexSet& iset = index[0]; 

   const Index_type array_length = last_indx + 1;

   //
   // Allocate "parent" array for traversal tests and initialize to...
   //
   Real_ptr parent;
   posix_memalign((void **)&parent, DATA_ALIGN, array_length*sizeof(Real_type)) ;

   for (Index_type i=0 ; i<array_length; ++i) {
      parent[i] = Real_type( rand() % 65536 );
   }

   //
   // Collect actual indices in index set for testing.
   //
   RAJAVec<Index_type> is_indices;
   getIndices(is_indices, index[0]);


   runMinReduceTests( parent, array_length,
                      iset, is_indices );

   runMaxReduceTests( parent, array_length,
                      iset, is_indices );

   runSumReduceTests( parent, array_length,
                      iset, is_indices );

   runMinLocReduceTests( parent, array_length,
                         iset, is_indices );

   runMaxLocReduceTests( parent, array_length,
                         iset, is_indices );


   ///
   /// Print total number of tests passed/run.
   ///
   cout << "\n All Tests : # passed / # run = " 
             << s_ntests_passed_total << " / " 
             << s_ntests_run_total << endl;


#if 0  // just screwing around with OpenMP

   int len = is_indices.size();
   vector<double> min_array(len);
   for (int j = 0; j < len; ++j) {
      min_array[j] = fabs( parent[ is_indices[j] ] );
   }
   const Index_type ref_min_indx = len/2;
   min_array[ref_min_indx] = ref_min_val;

   Real_type  ref_sum  = 0.0;

   for (Index_type i = 0; i < is_indices.size(); ++i) {
      ref_sum += parent[ is_indices[i] ];
   }
  
   double osum1 = 0.0; 
   double osum2 = 5.0;

   double omin1 = 1.0e+20;
   double omin2 = -200.0;

   #pragma omp parallel for reduction(+:osum1,osum2) reduction(min:omin1,omin2)
   for (int i = 0; i < len; ++i) {
      osum1 += parent[ is_indices[i] ]; 
      osum2 += 1.0;
      if ( min_array[ i ] < omin1 ) omin1 = min_array[ i ];
      if ( min_array[ i ] < omin2 ) omin2 = min_array[ i ];
   } 

   cout << "\n\nReduceSum OpenMP: osum1 = " << osum1 
             << " -- ( " << ref_sum << " )" << endl;
   cout << "ReduceSum OpenMP: osum2 = " << osum2 
             << " -- ( " << iset.getLength() + 5.0 << " )" << endl;
   cout << "ReduceMin OpenMP: omin1 = " << omin1 
             << " -- ( " << ref_min_val << " )" << endl;
   cout << "ReduceMin OpenMP: omin2 = " << omin2 
             << " -- ( " << -200.0 << " )" << endl;

#endif 

//
// Clean up....
//
   free(parent);

   cout << "\n DONE!!! " << endl;

   return 0 ;
}

