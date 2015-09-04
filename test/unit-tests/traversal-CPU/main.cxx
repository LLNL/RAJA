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

#include "buildIndexSet.hxx"


//
// Global variables for counting tests executed/passed.
//
unsigned s_ntests_run_total = 0;
unsigned s_ntests_passed_total = 0;

unsigned s_ntests_run = 0;
unsigned s_ntests_passed = 0;


//
//  Simple utility function to check results of forall traversals
//
void forall_CheckResult(const std::string& name,
                        Real_ptr ref_result,  
                        Real_ptr to_check,  
                        Index_type alen)
{
   s_ntests_run_total++;
   s_ntests_run++;
  
   bool is_correct = true;
   for (Index_type i = 0 ; i < alen && is_correct; ++i) {
      is_correct &= ref_result[i] == to_check[i];
   }

   if ( is_correct ) {
      s_ntests_passed_total++;
      s_ntests_passed++;
   } else {
      std::cout << name << " is WRONG" << std::endl; 
   }
}

//
//  Simple utility function to check results of forall_Icount traversals
//
void forall_Icount_CheckResult(const std::string& name,
                               Real_ptr ref_result,
                               Real_ptr to_check,
                               Index_type alen)
{
   s_ntests_run_total++;
   s_ntests_run++;

   bool is_correct = true;
   for (Index_type i = 0 ; i < alen && is_correct; ++i) {
      is_correct &= ref_result[i] == to_check[i];
   }

   if ( is_correct ) {
      s_ntests_passed_total++;
      s_ntests_passed++;
   } else {
      std::cout << name << " is WRONG" << std::endl;
   }
}

///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA traversal tests
// based on execution policy template
//
///////////////////////////////////////////////////////////////////////////
template <typename ISET_POLICY_T>
void runBasicForallTest(const std::string& policy,
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

   std::cout << "\n Test forall execution for " << policy << "\n";

   forall< ISET_POLICY_T >( iset, [=] (Index_type idx) {
      test_array[idx] = in_array[idx] * in_array[idx]; 
   } );

   forall_CheckResult(policy, ref_array, test_array, alen);

#if 0
   std::cout << std::endl << std::endl;
   for (Index_type i = 0; i < is_indices.size(); ++i) {
      std::cout << "test_array[" << is_indices[i] << "] = " 
                << test_array[ is_indices[i] ]
                << " ( " << ref_array[ is_indices[i] ] << " ) " << std::endl;
   }
   std::cout << std::endl;
#endif

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

   std::cout << "\n\n BEGIN RAJA::forall tests: ibuild = " 
             << ibuild << std::endl;

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

#if defined(RAJA_COMPILER_ICC)
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

   std::cout << "\n tests passed / test run: "
             << s_ntests_passed << " / " << s_ntests_run << std::endl;

   std::cout << "\n END RAJA::forall tests: ibuild = " 
             << ibuild << std::endl;
}


///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA Icount traversal tests
// based on execution policy template
//
///////////////////////////////////////////////////////////////////////////
template <typename ISET_POLICY_T>
void runBasicForall_IcountTest(const std::string& policy,
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

   std::cout << "\n Test forall_Icount execution for " << policy << "\n";

   forall_Icount< ISET_POLICY_T >( iset, [=] (Index_type icount, Index_type idx) {
      test_array[icount] = in_array[idx] * in_array[idx];
   } );

   forall_Icount_CheckResult(policy, ref_array, test_array, test_alen);

#if 0
   std::cout << std::endl << std::endl;
   for (Index_type i = 0; i < is_indices.size(); ++i) {
      std::cout << "test_array[" << i << "] = " << test_array[ i ]
                << " ( " << ref_array[ i ] << " ) " << std::endl;
   }
   std::cout << std::endl;
#endif

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

   std::cout << "\n\n BEGIN RAJA::forall_Icount tests: ibuild = " 
             << ibuild << std::endl;

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

   std::cout << "\n tests passed / test run: "
             << s_ntests_passed << " / " << s_ntests_run << std::endl;

   std::cout << "\n END RAJA::forall_Icount tests: ibuild = " 
             << ibuild << std::endl;
}



///////////////////////////////////////////////////////////////////////////
//
// Main Program.
//
///////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{

// 
//  All methods to construct index sets should generate equivalent results.
//
   IndexSet index[NumBuildMethods];
   for (unsigned ibuild = 0; ibuild < NumBuildMethods; ++ibuild) {
      buildIndexSet( index, static_cast<IndexSetBuildMethod>(ibuild) );
   }  

#if 0 
RDH TODO -- checks for equality of index set variants here (need to add proper 
            methods to IndexSet class) instead of running tests for each
            index set?
#endif

#if 0
   std::cout << std::endl << std::endl;
   for (unsigned ibuild = 0; ibuild < NumBuildMethods; ++ibuild) {
      std::cout << "index with build method " << ibuild << std::endl;
      index[ibuild].print(std::cout);
      std::cout << std::endl;
   } 
   std::cout << std::endl;
#endif

#if 0  // Test attempt to add invalid segment type.
   RangeStrideSegment rs_segment(0, 4, 2);
   if ( index[0].isValidSegmentType(&rs_segment) ) {
      std::cout << "RangeStrideSegment VALID for index[0]" << std::endl;
   } else {
      std::cout << "RangeStrideSegment INVALID for index[0]" << std::endl;
   }
   index[0].push_back(rs_segment);
   index[0].push_back_nocopy(&rs_segment);
#endif


   //
   // Allocate and initialize arrays for tests...
   //
   const Index_type array_length = 2000;

   Real_ptr parent;
   posix_memalign((void **)&parent, DATA_ALIGN, array_length*sizeof(Real_type)) ;

   for (Index_type i=0 ; i<array_length; ++i) {
      parent[i] = static_cast<Real_type>( rand() % 65536 );
   }


///////////////////////////////////////////////////////////////////////////
// Set up indexing information for tests...
///////////////////////////////////////////////////////////////////////////
   RAJAVec<Index_type> is_indices = getIndices(index[0]);


///////////////////////////////////////////////////////////////////////////
//
// Run RAJA::forall loop iteration tests...
//
///////////////////////////////////////////////////////////////////////////

   for (unsigned ibuild = 0; ibuild < NumBuildMethods; ++ibuild) {
      runForallTests( ibuild, parent, array_length, 
                      index[ibuild], is_indices );
   }

   for (unsigned ibuild = 0; ibuild < NumBuildMethods; ++ibuild) {
      runForall_IcountTests( ibuild, parent, array_length, 
                             index[ibuild], is_indices );
   }


#if 0 
///////////////////////////////////////////////////////////////////////////
//
// Check some basic conditional IndexSet construction operations....
//
///////////////////////////////////////////////////////////////////////////

#if !defined(RAJA_COMPILER_XLC12)

   RAJAVec<Index_type> even_indices; 
   RAJAVec<Index_type> lt_300_indices; 

   even_indices = 
      getIndicesConditional(index[0], [](Index_type idx) { return !(idx%2);} );

   lt_300_indices = 
      getIndicesConditional(index[0], [](Index_type idx) { return (idx<300);} );

   IndexSet hiset_even;
   hiset_even.push_back( ListSegment(&even_indices[0], even_indices.size()) );

   std::cout << "\n\n INDEX SET WITH EVEN INDICES ONLY..." << std::endl;
   hiset_even.print(std::cout);


   IndexSet hiset_lt_300;
   hiset_even.push_back( ListSegment(&lt_300_indices[0], lt_300_indices.size()) );

   std::cout << "\n\n INDEX SET WITH INDICES < 300 ONLY..." << std::endl;
   hiset_lt_300.print(std::cout);

#endif  //  !defined(RAJA_COMPILER_XLC12)

#endif  //  do basic conditional checks...

   ///
   /// Print number of tests passed/run.
   ///
   std::cout << "\n All Tests : # run / # passed = " 
             << s_ntests_passed_total << " / " 
             << s_ntests_run_total << std::endl;

//
// Clean up....
//
   free(parent);

   std::cout << "\n DONE!!! " << std::endl;

   return 0 ;
}

