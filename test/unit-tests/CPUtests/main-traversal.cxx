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

#include<string>
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


///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA traversal tests
// based on execution policy template
//
///////////////////////////////////////////////////////////////////////////
template <typename ISET_POLICY_T>
void runBasicForallTest(const string& policy,
                        Real_ptr in_array, Index_type alen,
                        const IndexSet& iset,
                        const RAJAVec<Index_type>& is_indices)
{
   Real_ptr test_array;
   Real_ptr ref_array;
   posix_memalign((void **)&test_array, DATA_ALIGN, alen*sizeof(Real_type)) ;
   posix_memalign((void **)&ref_array, DATA_ALIGN, alen*sizeof(Real_type)) ;

   for (Index_type i=0 ; i<alen; ++i) {
      test_array[i] = 0.0;
      ref_array[i] = 0.0;
   }

   //
   // Generate reference result to check correctness.
   // Note: Reference does not use RAJA!!!
   //
   for (Index_type i = 0; i < is_indices.size(); ++i) {
      ref_array[ is_indices[i] ] =
         in_array[ is_indices[i] ] * in_array[ is_indices[i] ];
   }

   cout << "\n Test forall execution for " << policy << "\n";

   s_ntests_run++;
   s_ntests_run_total++;

   forall< ISET_POLICY_T >( iset, [=] (Index_type idx) {
      test_array[idx] = in_array[idx] * in_array[idx]; 
   } );

   if ( !array_equal(ref_array, test_array, alen) ) {
      cout << "\n TEST FAILURE " << endl;
#if 0
      cout << endl << endl;
      for (Index_type i = 0; i < is_indices.size(); ++i) {
         cout << "test_array[" << is_indices[i] << "] = " 
                   << test_array[ is_indices[i] ]
                   << " ( " << ref_array[ is_indices[i] ] << " ) " << endl;
      }
      cout << endl;
#endif
   } else {
      s_ntests_passed++;
      s_ntests_passed_total++;
   }

   free(test_array);
   free(ref_array);
}

///////////////////////////////////////////////////////////////////////////
//
// Run RAJA::forall tests with available RAJA execution policies....
//
///////////////////////////////////////////////////////////////////////////
void runForallTests( unsigned ibuild, 
                     Real_ptr in_array,  Index_type alen,
                     const IndexSet& iset, 
                     const RAJAVec<Index_type>& is_indices )
{

   cout << "\n\n BEGIN RAJA::forall tests: ibuild = " 
             << ibuild << endl;

   // initialize test counters for this test set
   s_ntests_run = 0;
   s_ntests_passed = 0;

   runBasicForallTest<
      IndexSet::ExecPolicy<seq_segit, seq_exec> > (
               "ExecPolicy<seq_segit, seq_exec>",
                in_array, alen,
                iset, is_indices );

   runBasicForallTest<
      IndexSet::ExecPolicy<seq_segit, simd_exec> > (
               "ExecPolicy<seq_segit, simd_exec>",
                in_array, alen,
                iset, is_indices );

#if defined(RAJA_USE_OPENMP)
   runBasicForallTest<
      IndexSet::ExecPolicy<seq_segit, omp_parallel_for_exec> > (
               "ExecPolicy<seq_segit, omp_parallel_for_exec>",
                in_array, alen,
                iset, is_indices );

   runBasicForallTest<
      IndexSet::ExecPolicy<omp_parallel_for_segit, seq_exec> > (
               "ExecPolicy<omp_parallel_for_segit, seq_exec>",
                in_array, alen,
                iset, is_indices );

   runBasicForallTest<
      IndexSet::ExecPolicy<omp_parallel_for_segit, simd_exec> > (
               "ExecPolicy<omp_parallel_for_segit, simd_exec>",
                in_array, alen,
                iset, is_indices );
#endif

#if defined(RAJA_USE_CILK)
   runBasicForallTest<
      IndexSet::ExecPolicy<seq_segit, cilk_for_exec> > (
               "ExecPolicy<seq_segit, cilk_for_exec>",
                in_array, alen,
                iset, is_indices );

   runBasicForallTest<
      IndexSet::ExecPolicy<cilk_for_segit, seq_exec> > (
               "ExecPolicy<cilk_for_segit, seq_exec>",
                in_array, alen,
                iset, is_indices );

   runBasicForallTest<
      IndexSet::ExecPolicy<cilk_for_segit, simd_exec> > (
               "ExecPolicy<cilk_for_segit, simd_exec>",
                in_array, alen,
                iset, is_indices );
#endif

   cout << "\n tests passed / test run: "
             << s_ntests_passed << " / " << s_ntests_run << endl;

   cout << "\n END RAJA::forall tests: ibuild = " 
             << ibuild << endl;
}


///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA Icount traversal tests
// based on execution policy template
//
///////////////////////////////////////////////////////////////////////////
template <typename ISET_POLICY_T>
void runBasicForall_IcountTest(const string& policy,
                               Real_ptr in_array, Index_type alen,
                               const IndexSet& iset,
                               const RAJAVec<Index_type>& is_indices)
{
   Index_type test_alen = is_indices.size();
   Real_ptr test_array;
   Real_ptr ref_array;
   posix_memalign((void **)&test_array, DATA_ALIGN, test_alen*sizeof(Real_type)) ;
   posix_memalign((void **)&ref_array, DATA_ALIGN, test_alen*sizeof(Real_type)) ;

   for (Index_type i=0 ; i<test_alen; ++i) {
      test_array[i] = 0.0;
      ref_array[i] = 0.0;
   }

   //
   // Generate reference result to check correctness.
   // Note: Reference does not use RAJA!!!
   //
   for (Index_type i = 0; i < is_indices.size(); ++i) {
      ref_array[ i ] =
         in_array[ is_indices[i] ] * in_array[ is_indices[i] ];
   }

   cout << "\n Test forall_Icount execution for " << policy << "\n";

   s_ntests_run++;
   s_ntests_run_total++;

   forall_Icount< ISET_POLICY_T >( iset, [=] (Index_type icount, Index_type idx) {
      test_array[icount] = in_array[idx] * in_array[idx];
   } );


   if ( !array_equal(ref_array, test_array, test_alen) ) {
      cout << "\n TEST FAILURE " << endl;
#if 0
      cout << endl << endl;
      for (Index_type i = 0; i < test_alen; ++i) {
         cout << "test_array[" << i << "] = " << test_array[ i ]
              << " ( " << ref_array[ i ] << " ) " << endl;
      }
      cout << endl;
#endif
   } else {
      s_ntests_passed++;
      s_ntests_passed_total++;
   }


   free(test_array);
   free(ref_array);
}

///////////////////////////////////////////////////////////////////////////
//
// Run RAJA::forall_Icount tests with available RAJA execution policies....
//
///////////////////////////////////////////////////////////////////////////
void runForall_IcountTests( unsigned ibuild, 
                            Real_ptr in_array,  Index_type alen,
                            const IndexSet& iset, 
                            const RAJAVec<Index_type>& is_indices )
{

   cout << "\n\n BEGIN RAJA::forall_Icount tests: ibuild = " 
             << ibuild << endl;

   // initialize test counters for this test set
   s_ntests_run = 0;
   s_ntests_passed = 0;

   runBasicForall_IcountTest<
      IndexSet::ExecPolicy<seq_segit, seq_exec> > (
               "ExecPolicy<seq_segit, seq_exec>",
                in_array, alen,
                iset, is_indices );

   runBasicForall_IcountTest<
      IndexSet::ExecPolicy<seq_segit, simd_exec> > (
               "ExecPolicy<seq_segit, simd_exec>",
                in_array, alen,
                iset, is_indices );

#if defined(RAJA_USE_OPENMP)
   runBasicForall_IcountTest<
      IndexSet::ExecPolicy<seq_segit, omp_parallel_for_exec> > (
               "ExecPolicy<seq_segit, omp_parallel_for_exec>",
                in_array, alen,
                iset, is_indices );

   runBasicForall_IcountTest<
      IndexSet::ExecPolicy<omp_parallel_for_segit, seq_exec> > (
               "ExecPolicy<omp_parallel_for_segit, seq_exec>",
                in_array, alen,
                iset, is_indices );

   runBasicForall_IcountTest<
      IndexSet::ExecPolicy<omp_parallel_for_segit, simd_exec> > (
               "ExecPolicy<omp_parallel_for_segit, simd_exec>",
                in_array, alen,
                iset, is_indices );
#endif

#if defined(RAJA_COMPILER_ICC)
   runBasicForall_IcountTest<
      IndexSet::ExecPolicy<seq_segit, cilk_for_exec> > (
               "ExecPolicy<seq_segit, cilk_for_exec>",
                in_array, alen,
                iset, is_indices );

   runBasicForall_IcountTest<
      IndexSet::ExecPolicy<cilk_for_segit, seq_exec> > (
               "ExecPolicy<cilk_for_segit, seq_exec>",
                in_array, alen,
                iset, is_indices );

   runBasicForall_IcountTest<
      IndexSet::ExecPolicy<cilk_for_segit, simd_exec> > (
               "ExecPolicy<cilk_for_segit, simd_exec>",
                in_array, alen,
                iset, is_indices );
#endif

   cout << "\n tests passed / test run: "
             << s_ntests_passed << " / " << s_ntests_run << endl;

   cout << "\n END RAJA::forall_Icount tests: ibuild = " 
             << ibuild << endl;
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
// Run checks for equality of all constructed index sets.
//
///////////////////////////////////////////////////////////////////////////

   cout << "\n\n BEGIN IndexSet equality/inequality tests " << endl;

   // initialize test counters for this test set
   s_ntests_run = 0;
   s_ntests_passed = 0;

   for (unsigned ibuild = 1; ibuild < NumBuildMethods; ++ibuild) {

      s_ntests_run++;
      s_ntests_run_total++;

      if ( index[ibuild] == index[0] ) {
         s_ntests_passed++; 
         s_ntests_passed_total++; 
      } else {
         cout << "\tIndexSet " << ibuild 
              << " DOES NOT MATCH IndexSet 0!! " << endl;
      }

      s_ntests_run++;
      s_ntests_run_total++;

      if ( index[ibuild] != index[0] ) {
         cout << "\tIndexSet " << ibuild
              << " MATCHES IndexSet 0!! " << endl;
      } else {
         s_ntests_passed++; 
         s_ntests_passed_total++;
      }

#if 0
      cout << endl << endl << "index 0 " << endl << endl;
      index[ibuild].print(cout);
      cout << endl << endl;
      cout << "index with build method " << ibuild << endl;
      index[ibuild].print(cout);
      cout << endl << endl;
#endif

   }

   cout << "\n tests passed / test run: "
        << s_ntests_passed << " / " << s_ntests_run << endl;

   cout << "\n END IndexSet equality/inequality tests " << endl;


///////////////////////////////////////////////////////////////////////////
//
// Run checks for adding invalid segment type to index set.
//
///////////////////////////////////////////////////////////////////////////

   cout << "\n\n BEGIN IndexSet invalid segment tests " << endl;

   // initialize test counters for this test set
   s_ntests_run = 0;
   s_ntests_passed = 0;

   RangeStrideSegment rs_segment(0, 4, 2);
   s_ntests_run++;
   s_ntests_run_total++;
   if ( index[0].isValidSegmentType(&rs_segment) ) {
      cout << "RangeStrideSegment reported as VALID for index[0]!!!" << endl;
   } else {
      s_ntests_passed++;
      s_ntests_passed_total++;
   }

   s_ntests_run++;
   s_ntests_run_total++;
   if ( index[0].push_back(rs_segment) ) {
      cout << "push_back(RangeStrideSegment) SUCCEEDED!!!" << endl;
   } else {
      s_ntests_passed++;
      s_ntests_passed_total++;
   }

   s_ntests_run++;
   s_ntests_run_total++;
   if ( index[0].push_back_nocopy(&rs_segment) ) {
      cout << "push_back_cocopy(RangeStrideSegment) SUCCEEDED!!!" << endl;
   } else {
      s_ntests_passed++;
      s_ntests_passed_total++;
   }

   cout << "\n tests passed / test run: "
        << s_ntests_passed << " / " << s_ntests_run << endl;

   cout << "\n END IndexSet invalid segment tests " << endl;


///////////////////////////////////////////////////////////////////////////
//
// Run RAJA::forall loop iteration tests...
//
///////////////////////////////////////////////////////////////////////////

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

   // initialize test counters for this test set
   s_ntests_run = 0;
   s_ntests_passed = 0;

   //
   // Earlier tests should confirm all index sets are identical.
   // But, can still run tests for all index sets if desired.
   //
   unsigned run_tests = 1;
// unsigned run_tests = NumBuildMethods;

   for (unsigned ibuild = 0; ibuild < run_tests; ++ibuild) {
      runForallTests( ibuild, parent, array_length, 
                      index[ibuild], is_indices );
   }

   for (unsigned ibuild = 0; ibuild < run_tests; ++ibuild) {
      runForall_IcountTests( ibuild, parent, array_length, 
                             index[ibuild], is_indices );
   }


#if !defined(RAJA_COMPILER_XLC12) && 1

///////////////////////////////////////////////////////////////////////////
//
// Check some basic conditional IndexSet operations....
//
///////////////////////////////////////////////////////////////////////////

   cout << "\n\n BEGIN IndexSet conditional operation tests " << endl;

   // initialize test counters for this test set
   s_ntests_run = 0;
   s_ntests_passed = 0;


   s_ntests_run++;
   s_ntests_run_total++;

   RAJAVec<Index_type> even_indices; 
   getIndicesConditional(even_indices, index[0], 
                         [](Index_type idx) { return !(idx%2);} );

   RAJAVec<Index_type> ref_even_indices;
   for (Index_type i = 0; i < is_indices.size(); ++i ) {
      Index_type idx = is_indices[i];
      if ( idx % 2 == 0 ) ref_even_indices.push_back(idx); 
   }

   if ( ( even_indices.size() == ref_even_indices.size() ) &&
        array_equal(&even_indices[0], &ref_even_indices[0], 
                    ref_even_indices.size()) ) {
      s_ntests_passed++;
      s_ntests_passed_total++;
   } else {
      cout << "\n even_indices TEST_FAILURE " << endl;
   }

// -------------------------------------------------------------------------   

   s_ntests_run++;
   s_ntests_run_total++;

   RAJAVec<Index_type> lt300_indices; 
   getIndicesConditional(lt300_indices, index[0],
                         [](Index_type idx) { return (idx<300);} );

   RAJAVec<Index_type> ref_lt300_indices;
   for (Index_type i = 0; i < is_indices.size(); ++i ) {
      Index_type idx = is_indices[i];
      if ( idx < 300 ) ref_lt300_indices.push_back(idx);
   }

   if ( ( lt300_indices.size() == ref_lt300_indices.size() ) &&
        array_equal(&lt300_indices[0], &ref_lt300_indices[0], 
                    ref_lt300_indices.size()) ) {
      s_ntests_passed++;
      s_ntests_passed_total++;
   } else {
      cout << "\n lt300_indices TEST_FAILURE " << endl;
   }

   cout << "\n tests passed / test run: "
             << s_ntests_passed << " / " << s_ntests_run << endl;

   cout << "\n END IndexSet conditional operation tests " << endl;

#endif  //  !defined(RAJA_COMPILER_XLC12) || 0

   ///
   /// Print number of tests passed/run.
   ///
   cout << "\n All Tests : # passed / # run = " 
             << s_ntests_passed_total << " / " 
             << s_ntests_run_total << endl;

//
// Clean up....
//
   free(parent);

   cout << "\n DONE!!! " << endl;

   return 0 ;
}

