//
// Main program illustrating simple RAJA index set creation 
// and execution and methods.
//

#include <cstdlib>
#include <time.h>

#include<string>
#include<vector>
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
//  Simple utility function to check results of min/maxloc reductions
//
void forall_reduceloc_CheckResult(const std::string& name,
                                  const Real_type ref_val,
                                  const Index_type ref_idx,
                                  Real_type check_val,
                                  Index_type check_idx)
{
   s_ntests_run_total++;
   s_ntests_run++;
  
   bool is_correct = (ref_val == check_val) && (ref_idx == check_idx);

   if ( is_correct ) {
      s_ntests_passed_total++;
      s_ntests_passed++;
   } else {
      std::cout << name << " is WRONG" << std::endl;                               
   } 
}

//
//  Simple utility function to check results of min/maxloc reductions
//
void forall_reduce_CheckResult(const std::string& name,
                               const Real_type ref_val,
                               Real_type check_val)
{
   s_ntests_run_total++;
   s_ntests_run++;
  
   bool is_correct = (ref_val == check_val);

   if ( is_correct ) {
      s_ntests_passed_total++;
      s_ntests_passed++;
   } else {
      std::cout << name << " is WRONG" << std::endl;
   } 
}

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
void runBasicMinReductionTest(const std::string& policy,
                              Real_ptr in_array, Index_type alen,
                              const IndexSet& iset,
                              RAJAVec<Index_type> is_indices)
{
   Real_ptr test_array;
   posix_memalign((void **)&test_array, DATA_ALIGN, alen*sizeof(Real_type)) ;

   //
   // Make all test array values positve
   //
   for (Index_type i=0 ; i<alen; ++i) {
      test_array[i] = std::abs( in_array[i] );
   }

   //
   // Generate reference result for min in middle of index set.
   //
   const Index_type ref_min_indx =
      static_cast<Index_type>(is_indices[is_indices.size()/2]);
   const Real_type  ref_min_val  = -100.0;

   test_array[ref_min_indx] = ref_min_val;

#if 0
   std::cout << "ref_min_indx = " << ref_min_indx << std::endl;
   std::cout << "ref_min_val = " << ref_min_val << std::endl;
   std::cout << "test_array[ref_min_indx] = " 
             << test_array[ref_min_indx] << std::endl;
#endif 

   std::cout << "\n Test MIN reduction for " << policy << "\n";

   ReduceMin<REDUCE_POLICY_T, Real_type> tmin0(1.0e+20);
   ReduceMin<REDUCE_POLICY_T, Real_type> tmin1(-200.0);

   int loops = 2;

   for (int k = 1; k <= loops; ++ k) {

//    std::cout << "k = " << k << std::endl;

      forall< ISET_POLICY_T >( iset, [=] (Index_type idx) {
         tmin0.min(k*test_array[idx]);
         tmin1.min(test_array[idx]);
      } );

      forall_reduce_CheckResult("ReduceMin:" + policy + ": tmin0",
                                k*ref_min_val, tmin0);
      forall_reduce_CheckResult("ReduceMin:" + policy + ": tmin1",
                                -200.0, tmin1);

#if 0
      std::cout << "tmin0 = " <<  static_cast<Real_type>(tmin0) 
                              << " -- ( " << k*ref_min_val << " ) " << std::endl;
      std::cout << "tmin1 = " <<  static_cast<Real_type>(tmin1) 
                              << " -- ( " << -200.0 << " ) " << std::endl;
#endif
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
   std::cout << "\n\n   BEGIN RAJA::forall MIN REDUCE tests...." << std::endl;

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

   std::cout << "\n tests passed / test run: " 
             << s_ntests_passed << " / " << s_ntests_run << std::endl; 

   std::cout << "\n   END RAJA::forall MIN REDUCE tests... " << std::endl;
}


///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA max reduction tests 
// based on execution policy template
//
///////////////////////////////////////////////////////////////////////////
template <typename ISET_POLICY_T,
          typename REDUCE_POLICY_T>
void runBasicMaxReductionTest(const std::string& policy,
                              Real_ptr in_array, Index_type alen,
                              const IndexSet& iset,
                              RAJAVec<Index_type> is_indices)
{
   Real_ptr test_array;
   posix_memalign((void **)&test_array, DATA_ALIGN, alen*sizeof(Real_type)) ;

   //
   // Make all test array values negative
   //
   for (Index_type i=0 ; i<alen; ++i) {
      test_array[i] = -std::abs( in_array[i] );
   }

   //
   // Generate reference result for max in middle of index set.
   //
   const Index_type ref_max_indx =
      static_cast<Index_type>(is_indices[is_indices.size()/2]);
   const Real_type  ref_max_val  = 100.0;

   test_array[ref_max_indx] = ref_max_val;

#if 0
   std::cout << "ref_max_indx = " << ref_max_indx << std::endl;
   std::cout << "ref_max_val = " << ref_max_val << std::endl;
   std::cout << "test_array[ref_max_indx] = " 
             << test_array[ref_max_indx] << std::endl;
#endif 

   std::cout << "\n Test MAX reduction for " << policy << "\n";

   ReduceMax<REDUCE_POLICY_T, Real_type> tmax0(-1.0e+20);
   ReduceMax<REDUCE_POLICY_T, Real_type> tmax1(200.0);

   int loops = 2;

   for (int k = 1; k <= loops; ++ k) {

//    std::cout << "k = " << k << std::endl;

      forall< ISET_POLICY_T >( iset, [=] (Index_type idx) {
         tmax0.max(k*test_array[idx]);
         tmax1.max(test_array[idx]);
      } );

      forall_reduce_CheckResult("ReduceMax:" + policy + ": tmax0",
                                k*ref_max_val, tmax0);
      forall_reduce_CheckResult("ReduceMin:" + policy + ": tmax1",
                                200.0, tmax1);
#if 0
      std::cout << "tmax0 = " <<  static_cast<Real_type>(tmax0) 
                              << " -- ( " << k*ref_max_val << " ) " << std::endl;
      std::cout << "tmax1 = " <<  static_cast<Real_type>(tmax1) 
                              << " -- ( " << 200.0 << " ) " << std::endl;
#endif
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
   std::cout << "\n\n   BEGIN RAJA::forall MAX REDUCE tests...." << std::endl;

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

   std::cout << "\n tests passed / test run: " 
             << s_ntests_passed << " / " << s_ntests_run << std::endl; 
   

   std::cout << "\n   END RAJA::forall MAX REDUCE tests... " << std::endl;
}


///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA sum reduction tests 
// based on execution policy template
//
///////////////////////////////////////////////////////////////////////////
template <typename ISET_POLICY_T,
          typename REDUCE_POLICY_T>
void runBasicSumReductionTest(const std::string& policy,
                              Real_ptr in_array, Index_type alen,
                              const IndexSet& iset,
                              RAJAVec<Index_type> is_indices)
{
   //
   // Generate reference result for sum
   //
   Real_type  ref_sum  = 0.0;

   for (Index_type i = 0; i < is_indices.size(); ++i) {
      ref_sum += in_array[ is_indices[i] ];
   }

#if 0
   std::cout << "ref_sum = " << ref_sum << std::endl;
#endif 

   std::cout << "\n Test SUM reduction for " << policy << "\n";

   ReduceSum<REDUCE_POLICY_T, Real_type> tsum0(0.0);
   ReduceSum<REDUCE_POLICY_T, Real_type> tsum1(5.0);

   int loops = 2;

   for (int k = 1; k <= loops; ++ k) {

//    std::cout << "k = " << k << std::endl;

      forall< ISET_POLICY_T >( iset, [=] (Index_type idx) {
         tsum0 += in_array[idx];
         tsum1 += 1.0;
      } );

      forall_reduce_CheckResult("ReduceSum:" + policy + ": tsum0",
                                k*ref_sum, tsum0);
      forall_reduce_CheckResult("ReduceMin:" + policy + ": tsum1",
                                k*iset.getLength() + 5.0, tsum1);
#if 0
      std::cout << "tsum0 = " <<  static_cast<Real_type>(tsum0) 
                              << " -- ( " << k*ref_sum << " ) " << std::endl;
      std::cout << "tmax1 = " <<  static_cast<Real_type>(tsum1) 
                              << " -- ( " << k*iset.getLength() + 5.0 
                              << " ) " << std::endl;
#endif
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
   std::cout << "\n\n   BEGIN RAJA::forall SUM REDUCE tests...." << std::endl;

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

   std::cout << "\n tests passed / test run: " 
             << s_ntests_passed << " / " << s_ntests_run << std::endl; 
   

   std::cout << "\n   END RAJA::forall SUM REDUCE tests... " << std::endl;
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
   IndexSet iset;
   buildIndexSet( iset );

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
   RAJAVec<Index_type> is_indices = getIndices(iset);


///////////////////////////////////////////////////////////////////////////
//
// Run RAJA reduction tests...
//
///////////////////////////////////////////////////////////////////////////

   runMinReduceTests( parent, array_length,
                      iset, is_indices );

   runMaxReduceTests( parent, array_length,
                      iset, is_indices );

   runSumReduceTests( parent, array_length,
                      iset, is_indices );


   ///
   /// Print total number of tests passed/run.
   ///
   std::cout << "\n All Tests : " 
             << s_ntests_passed_total << " / " 
             << s_ntests_run_total << std::endl;


#if 0  // just screwing around with OpenMP

   int len = is_indices.size();
   std::vector<double> min_array(len);
   for (int j = 0; j < len; ++j) {
      min_array[j] = std::abs( parent[ is_indices[j] ] );
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

   std::cout << "\n\nReduceSum OpenMP: osum1 = " << osum1 
             << " -- ( " << ref_sum << " )" << std::endl;
   std::cout << "ReduceSum OpenMP: osum2 = " << osum2 
             << " -- ( " << iset.getLength() + 5.0 << " )" << std::endl;
   std::cout << "ReduceMin OpenMP: omin1 = " << omin1 
             << " -- ( " << ref_min_val << " )" << std::endl;
   std::cout << "ReduceMin OpenMP: omin2 = " << omin2 
             << " -- ( " << -200.0 << " )" << std::endl;

#endif 

//
// Clean up....
//
   free(parent);
   free(child);
   free(child_ref);

   std::cout << "\n DONE!!! " << std::endl;

   return 0 ;
}

