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
unsigned ntests_run_total = 0;
unsigned ntests_passed_total = 0;

unsigned ntests_run = 0;
unsigned ntests_passed = 0;


//
//  Simple utility function to check results of forall traversals
//
void forall_CheckResult(const std::string& name,
                        Real_ptr ref_result,  
                        Real_ptr to_check,  
                        Index_type iset_len)
{
   ntests_run_total++;
   ntests_run++;
  
   bool is_correct = true;
   for (Index_type i = 0 ; i < iset_len && is_correct; ++i) {
      is_correct &= ref_result[i] == to_check[i];
   }

   if ( is_correct ) {
      ntests_passed_total++;
      ntests_passed++;
   } else {
      std::cout << name << " is WRONG" << std::endl; 
   }
}


///////////////////////////////////////////////////////////////////////////
//
// Run RAJA::forall tests with available RAJA execution policies....
//
///////////////////////////////////////////////////////////////////////////
void runForAllTests( unsigned ibuild, 
                     const IndexSet& hindex, 
                     Real_ptr parent, 
                     Real_ptr child, 
                     Real_ptr child_ref )
{
   auto forall_op = [&] (Index_type idx) {
                       child[idx] = parent[idx] * parent[idx]; };

   std::cout << "\n\n BEGIN RAJA::forall tests: ibuild = " 
             << ibuild << std::endl;

   forall< IndexSet::ExecPolicy<seq_segit, seq_exec> >(hindex, forall_op);
   forall_CheckResult("IndexSet::ExecPolicy<seq_segit, seq_exec>",
                      child_ref, child, hindex.getLength());

   forall< IndexSet::ExecPolicy<seq_segit, simd_exec> >(hindex, forall_op);
   forall_CheckResult("IndexSet::ExecPolicy<seq_segit, simd_exec>",
                      child_ref, child, hindex.getLength());

   forall< IndexSet::ExecPolicy<seq_segit, omp_parallel_for_exec> >(hindex, forall_op);
   forall_CheckResult("IndexSet::ExecPolicy<seq_segit, omp_parallel_for_exec>",
                      child_ref, child, hindex.getLength());

   forall< IndexSet::ExecPolicy<omp_parallel_for_segit, seq_exec> >(hindex, forall_op);
   forall_CheckResult("IndexSet::ExecPolicy<omp_parallel_for_segit, seq_exec>",
                      child_ref, child, hindex.getLength());

   forall< IndexSet::ExecPolicy<omp_parallel_for_segit, simd_exec> >(hindex, forall_op);
   forall_CheckResult("IndexSet::ExecPolicy<omp_parallel_for_segit, simd_exec>",
                      child_ref, child, hindex.getLength());

#if defined(RAJA_COMPILER_ICC)
   forall< IndexSet::ExecPolicy<seq_segit, cilk_for_exec> >(hindex, forall_op);
   forall_CheckResult("IndexSet::ExecPolicy<seq_segit, cilk_for_exec>",
                      child_ref, child, hindex.getLength());

   forall< IndexSet::ExecPolicy<cilk_for_segit, seq_exec> >(hindex, forall_op);
   forall_CheckResult("IndexSet::ExecPolicy<cilk_for_segit, seq_exec>",
                      child_ref, child, hindex.getLength());

   forall< IndexSet::ExecPolicy<cilk_for_segit, simd_exec> >(hindex, forall_op);
   forall_CheckResult("IndexSet::ExecPolicy<cilk_for_segit, simd_exec>",
                      child_ref, child, hindex.getLength());

   forall< IndexSet::ExecPolicy<cilk_for_segit, omp_parallel_for_exec> >(hindex, forall_op);
   forall_CheckResult("IndexSet::ExecPolicy<cilk_for_segit, omp_parallel_for_exec>",
                      child_ref, child, hindex.getLength());

   forall< IndexSet::ExecPolicy<omp_parallel_for_segit, cilk_for_exec> >(hindex, forall_op);
   forall_CheckResult("IndexSet::ExecPolicy<omp_parallel_for_segit, cilk_for_exec>",
                      child_ref, child, hindex.getLength());
#endif

#if 0 // print output for manual checking...
   std::cout << "\n CHILD ARRAY OUTPUT... " << std::endl;

   for (Index_type ic = 0; ic < array_length; ++ic) {
      std::cout << "child[" << ic << "] = " << child[ic] << std::endl; ;
   }
#endif

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
   IndexSet hindex[NumBuildMethods];
   for (unsigned ibuild = 0; ibuild < NumBuildMethods; ++ibuild) {
      buildIndexSet( hindex, static_cast<IndexSetBuildMethod>(ibuild) );
   }  

#if 0 
RDH TODO -- checks for equality here....when proper methods are added 
            to IndexSet class 
#endif

#if 0
   std::cout << std::endl << std::endl;
   for (unsigned ibuild = 0; ibuild < NumBuildMethods; ++ibuild) {
      std::cout << "hindex with build method " << ibuild << std::endl;
      hindex[ibuild].print(std::cout);
      std::cout << std::endl;
   } 
   std::cout << std::endl;
#endif

#if 1  // Test attempt to add invalid segment type.
   RangeStrideSegment rs_segment(0, 4, 2);
   if ( hindex[0].isValidSegmentType(&rs_segment) ) {
      std::cout << "RangeStrideSegment VALID for hindex[0]" << std::endl;
   } else {
      std::cout << "RangeStrideSegment INVALID for hindex[0]" << std::endl;
   }
   hindex[0].push_back(rs_segment);
   hindex[0].push_back_nocopy(&rs_segment);
#endif


   //
   // Allocate and initialize arrays for tests...
   //
   const Index_type array_length = 2000;

   Real_ptr parent;
   Real_ptr child;
   Real_ptr child_ref;
   posix_memalign((void **)&parent, DATA_ALIGN, array_length*sizeof(Real_type)) ;
   posix_memalign((void **)&child, DATA_ALIGN, array_length*sizeof(Real_type)) ;
   posix_memalign((void **)&child_ref, DATA_ALIGN, array_length*sizeof(Real_type)) ;

   for (Index_type i=0 ; i<array_length; ++i) {
      parent[i] = static_cast<Real_type>( rand() % 65536 );
      child[i] = 0.0;
      child_ref[i] = 0.0;
   }


///////////////////////////////////////////////////////////////////////////
// Set up indexing information for tests...
///////////////////////////////////////////////////////////////////////////
   RAJAVec<Index_type> is_indices = getIndices(hindex[0]);


///////////////////////////////////////////////////////////////////////////
//
// Run RAJA::forall loop iteration tests...
//
///////////////////////////////////////////////////////////////////////////

   //
   // Generate reference result to check correctness.
   // Note: Reference does not use RAJA!!!
   //
   for (Index_type i = 0; i < is_indices.size(); ++i) {
      child_ref[ is_indices[i] ] = 
         parent[ is_indices[i] ] * parent[ is_indices[i] ];
   }


   for (unsigned ibuild = 0; ibuild < NumBuildMethods; ++ibuild) {
      ntests_run = 0; 
      ntests_passed = 0; 

      runForAllTests( ibuild, hindex[ibuild], parent, child, child_ref );

      std::cout << "\n forall(ibuild = " << ibuild << ") : "
                << ntests_passed << " / " << ntests_run << std::endl; 
   }


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
      getIndicesConditional(hindex[0], [](Index_type idx) { return !(idx%2);} );

   lt_300_indices = 
      getIndicesConditional(hindex[0], [](Index_type idx) { return (idx<300);} );

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
             << ntests_passed_total << " / " << ntests_run_total << std::endl;

//
// Clean up....
//
   free(parent);
   free(child);
   free(child_ref);

   std::cout << "\n DONE!!! " << std::endl;

   return 0 ;
}

