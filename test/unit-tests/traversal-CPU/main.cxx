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
                        RAJAVec<Index_type> is_indices)
{
   Real_ptr test_array;
   Real_ptr ref_array;
   posix_memalign((void **)&test_array, DATA_ALIGN, alen*sizeof(Real_type)) ;
   posix_memalign((void **)&ref_array, DATA_ALIGN, alen*sizeof(Real_type)) ;

   for (Index_type i=0 ; i<array_length; ++i) {
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

   auto forall_op = [&] (Index_type idx) {
                       child[idx] = parent[idx] * parent[idx]; };

   free(test_array);
   free(ref_array);
}


///////////////////////////////////////////////////////////////////////////
//
// Run RAJA::forall tests with available RAJA execution policies....
//
///////////////////////////////////////////////////////////////////////////
void runForAllTests( unsigned ibuild, 
                     Real_ptr in_array,  Index_type alen,
                     const IndexSet& index, 
                     RAJAVec<Index_type> is_indices )
{

   std::cout << "\n\n BEGIN RAJA::forall tests: ibuild = " 
             << ibuild << std::endl;

   // initialize test counters for this test set
   s_ntests_run = 0;
   s_ntests_passed = 0;

   runBasicForallTest<
      IndexSet::ExecPolicy<seq_segit, seq_exec>, seq_reduce > (
               "ExecPolicy<seq_segit, seq_exec>",
                in_array, alen,
                iset, is_indices );

   std::cout << "\n tests passed / test run: "
             << s_ntests_passed << " / " << s_ntests_run << std::endl;

   std::cout << "\n END RAJA::forall tests: ibuild = " 
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
RDH TODO -- checks for equality here....when proper methods are added 
            to IndexSet class 
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

#if 1  // Test attempt to add invalid segment type.
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

#if 0
   //
   // Generate reference result to check correctness.
   // Note: Reference does not use RAJA!!!
   //
   for (Index_type i = 0; i < is_indices.size(); ++i) {
      child_ref[ is_indices[i] ] = 
         parent[ is_indices[i] ] * parent[ is_indices[i] ];
   }


   for (unsigned ibuild = 0; ibuild < NumBuildMethods; ++ibuild) {
      s_ntests_run = 0; 
      s_ntests_passed = 0; 

      runForAllTests( ibuild, index[ibuild], parent, child, child_ref );

      std::cout << "\n forall(ibuild = " << ibuild << ") TESTS : "
                << s_ntests_passed << " / " << s_ntests_run << std::endl; 
   }

#else
   
   for (unsigned ibuild = 0; ibuild < NumBuildMethods; ++ibuild) {
      runForAllTests( ibuild, parent, array_length, 
                      index[ibuild], is_indices );
   }

#endif


#if 0 
///////////////////////////////////////////////////////////////////////////
//
// Check some basic conditional IndexSet construction operations....
//
///////////////////////////////////////////////////////////////////////////

#if !defined(RAJA_COMPILER_XLC12)

#if defined(RAJA_USE_STL)
   std::vector<Index_type> even_indices;
   std::vector<Index_type> lt_300_indices;
#else
   RAJAVec<Index_type> even_indices; 
   RAJAVec<Index_type> lt_300_indices; 
#endif

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
   std::cout << "\n All Tests : " 
             << s_ntests_passed_total << " / " 
             << s_ntests_run_total << std::endl;

//
// Clean up....
//
   free(parent);
   free(child);
   free(child_ref);

   std::cout << "\n DONE!!! " << std::endl;

   return 0 ;
}

