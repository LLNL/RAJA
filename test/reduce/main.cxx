//
// Main program illustrating simple RAJA index set creation 
// and execution and methods.
//
// NOTE: Some compilers do not support C++ lambda expressions; for such
//       compilers, the macro constant USE_LAMBDA must be undefined to
//       compile and run program.
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
// For tested platforms, only xlc compiler fails to support C++ lambda fcns.
//
#define USE_LAMBDA



//
//  Simple utility function to check results of min/maxloc reductions
//
void forall_reduceloc_CheckResult(const std::string& name,
                                  const Real_type ref_val,
                                  const Index_type ref_idx,
                                  Real_type check_val,
                                  Index_type check_idx)
{
   ntests_run_total++;
   ntests_run++;
  
   bool is_correct = (ref_val == check_val) && (ref_idx == check_idx);

   if ( is_correct ) {
      ntests_passed_total++;
      ntests_passed++;
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
   ntests_run_total++;
   ntests_run++;
  
   bool is_correct = (ref_val == check_val);

   if ( is_correct ) {
      ntests_passed_total++;
      ntests_passed++;
   } else {
      std::cout << name << " is WRONG" << std::endl;
   } 
}


 
///////////////////////////////////////////////////////////////////////////
//
// Run RAJA::forall_min tests with available RAJA execution policies....
//
///////////////////////////////////////////////////////////////////////////
void runForAllMinTests( const IndexSet& hindex, 
                        Real_ptr parent, 
                        Real_type ref_min )
{
// Run various min reductions and check results against reference...

   std::cout << "\n\n BEGIN RAJA::forall MIN tests...." << std::endl;

{
   ReduceMin<seq_reduce, Real_type> tmin0(1.0e+20);
   ReduceMin<seq_reduce, Real_type> tmin1(-200.0);

   forall< IndexSet::ExecPolicy<seq_segit, seq_exec> >( hindex,
      [=] (Index_type idx) {
      tmin0.min(parent[idx]);
      tmin1.min(parent[idx]);
   } );
   forall_reduce_CheckResult("ExecPolicy<seq_segit, seq_exec>",
                             ref_min, tmin0);
   std::cout << "\n\t ExecPolicy<seq_segit, seq_exec> test:\n "
             << "\t result = " << tmin0
             << " -- ref result = "<< ref_min 
             << std::endl;

   std::cout << "\t tmin1 result = " << tmin1 << std::endl;
}

{
   ReduceMin<omp_reduce, Real_type> tmin0(1.0e+20);
   ReduceMin<seq_reduce, Real_type> tmin1(-200.0);

   forall< IndexSet::ExecPolicy<seq_segit, omp_parallel_for_exec> >( hindex,
      [=] (Index_type idx) {
      tmin0.min(parent[idx]);
      tmin1.min(parent[idx]);
   } );
   forall_reduce_CheckResult("ExecPolicy<seq_segit, omp_parallel_for_exec>",
                             ref_min, tmin0);
   std::cout << "\n\t ExecPolicy<seq_segit, omp_parallel_for_exec> test:\n"
             << "\t result = " << tmin0
             << " -- ref result = "<< ref_min 
             << std::endl;

   std::cout << "\t tmin1 result = " << tmin1 << std::endl;
}

#if 0
   tmin = 1.0e+20 ;
   tloc = -1 ;
   forall_minloc< IndexSet::ExecPolicy<seq_segit, simd_exec> >(
                                      hindex, &tmin, &tloc, forall_minloc_op);
   forall_reduceloc_CheckResult("IndexSet::ExecPolicy<seq_segit, simd_exec>",
                                ref_min_val, ref_min_indx, tmin, tloc);

   tmin = 1.0e+20 ;
   tloc = -1 ;
   forall_minloc< IndexSet::ExecPolicy<omp_parallel_for_segit, seq_exec> >(
                                      hindex, &tmin, &tloc, forall_minloc_op);
   forall_reduceloc_CheckResult("IndexSet::ExecPolicy<omp_parallel_for_segit, seq_exec>",
                                ref_min_val, ref_min_indx, tmin, tloc);

   tmin = 1.0e+20 ;
   tloc = -1 ;
   forall_minloc< IndexSet::ExecPolicy<omp_parallel_for_segit, simd_exec> >(
                                      hindex, &tmin, &tloc, forall_minloc_op);
   forall_reduceloc_CheckResult("IndexSet::ExecPolicy<omp_parallel_for_segit, simd_exec>",
                                ref_min_val, ref_min_indx, tmin, tloc);

#if defined(RAJA_COMPILER_ICC)
   tmin = 1.0e+20 ;
   tloc = -1 ;
   forall_minloc< IndexSet::ExecPolicy<seq_segit, cilk_for_exec> >(
                                      hindex, &tmin, &tloc, forall_minloc_op);
   forall_reduceloc_CheckResult("IndexSet::ExecPolicy<seq_segit, cilk_for_exec>",
                                ref_min_val, ref_min_indx, tmin, tloc);

   tmin = 1.0e+20 ;
   tloc = -1 ;
   forall_minloc< IndexSet::ExecPolicy<cilk_for_segit, seq_exec> >(
                                      hindex, &tmin, &tloc, forall_minloc_op);
   forall_reduceloc_CheckResult("IndexSet::ExecPolicy<cilk_for_segit, seq_exec>",
                                ref_min_val, ref_min_indx, tmin, tloc);

   tmin = 1.0e+20 ;
   tloc = -1 ;
   forall_minloc< IndexSet::ExecPolicy<cilk_for_segit, simd_exec> >(
                                      hindex, &tmin, &tloc, forall_minloc_op);
   forall_reduceloc_CheckResult("IndexSet::ExecPolicy<cilk_for_segit, simd_exec>",
                                ref_min_val, ref_min_indx, tmin, tloc);

   tmin = 1.0e+20 ;
   tloc = -1 ;
   forall_minloc< IndexSet::ExecPolicy<cilk_for_segit, omp_parallel_for_exec> >(
                                      hindex, &tmin, &tloc, forall_minloc_op);
   forall_reduceloc_CheckResult("IndexSet::ExecPolicy<cilk_for_segit, omp_parallel_for_exec>",
                                ref_min_val, ref_min_indx, tmin, tloc);

   tmin = 1.0e+20 ;
   tloc = -1 ;
   forall_minloc< IndexSet::ExecPolicy<omp_parallel_for_segit, cilk_for_exec> >(
                                      hindex, &tmin, &tloc, forall_minloc_op);
   forall_reduceloc_CheckResult("IndexSet::ExecPolicy<omp_parallel_for_segit, cilk_for_exec>",
                                ref_min_val, ref_min_indx, tmin, tloc);
#endif

#endif

   std::cout << "\n END RAJA::forall MIN tests... " << std::endl;
}


///////////////////////////////////////////////////////////////////////////
//
// Run RAJA::forall_sum tests with available RAJA execution policies....
//
///////////////////////////////////////////////////////////////////////////
void runForAllSumTests( const IndexSet& hindex, 
                        Real_ptr parent, 
                        Real_type ref_sum )
{
// Run various sum reductions and check results against reference...

   std::cout << "\n\n BEGIN RAJA::forall SUM tests...." << std::endl;

{

   ReduceSum<seq_reduce, Real_type> tsum0(0.0);
   ReduceSum<seq_reduce, Real_type> tsum1(5.0);

   forall< IndexSet::ExecPolicy<seq_segit, seq_exec> >( hindex,
      [=] (Index_type idx) {
      Real_type tval = parent[idx];
      tsum0 += parent[idx];
      parent[idx] = 1.0;
      tsum1 += parent[idx];
      parent[idx] = tval;
   } );

   forall_reduce_CheckResult("ExecPolicy<seq_segit, seq_exec>",
                             ref_sum, tsum0);
   std::cout << "\n\t ExecPolicy<seq_segit, seq_exec> test:\n "
             << "\t result = " << tsum0
             << " -- ref result = "<< ref_sum 
             << std::endl;

   std::cout << "\t tsum1 result = " << tsum1 
             << " -- ref result = "<< hindex.getLength() + 5.0 << std::endl;
}

{
   ReduceSum<omp_reduce, Real_type> tsum0(0.0);
   ReduceSum<seq_reduce, Real_type> tsum1(5.0);

   forall< IndexSet::ExecPolicy<seq_segit, seq_exec> >( hindex,
      [=] (Index_type idx) {
      Real_type tval = parent[idx];
      tsum0 += parent[idx];
      parent[idx] = 1.0;
      tsum1 += parent[idx];
      parent[idx] = tval;
   } );

   forall_reduce_CheckResult("ExecPolicy<seq_segit, omp_parallel_for_exec>",
                             ref_sum, tsum0);
   std::cout << "\n\t ExecPolicy<seq_segit, omp_parallel_for_exec> test:\n "
             << "\t result = " << tsum0
             << " -- ref result = "<< ref_sum 
             << std::endl;

   std::cout << "\t tsum1 result = " << tsum1 
             << " -- ref result = "<< hindex.getLength() + 5.0 << std::endl;
}

#if 0
// Run various maxloc traversals and check results against reference...

   Real_type tsum = 0.0;

   std::cout << "\n\n BEGIN RAJA::forall_sum tests: ibuild = " 
             << ibuild << std::endl;

   forall_sum< IndexSet::ExecPolicy<seq_segit, seq_exec> >(
                                   hindex, &tsum, forall_sum_op);
   forall_reduce_CheckResult("IndexSet::ExecPolicy<seq_segit, seq_exec>",
                             ref_sum, tsum);

   tsum = 0.0;
   forall_sum< IndexSet::ExecPolicy<seq_segit, simd_exec> >(
                                   hindex, &tsum, forall_sum_op);
   forall_reduce_CheckResult("IndexSet::ExecPolicy<seq_segit, simd_exec>",
                             ref_sum, tsum);

   tsum = 0.0;
   forall_sum< IndexSet::ExecPolicy<seq_segit, omp_parallel_for_exec> >(
                                   hindex, &tsum, forall_sum_op);
   forall_reduce_CheckResult("IndexSet::ExecPolicy<seq_segit, omp_parallel_for_exec>",
                             ref_sum, tsum);

   tsum = 0.0;
   forall_sum< IndexSet::ExecPolicy<omp_parallel_for_segit, seq_exec> >(
                                   hindex, &tsum, forall_sum_op);
   forall_reduce_CheckResult("IndexSet::ExecPolicy<omp_parallel_for_segit, seq_exec>",
                             ref_sum, tsum);

   tsum = 0.0;
   forall_sum< IndexSet::ExecPolicy<omp_parallel_for_segit, simd_exec> >(
                                   hindex, &tsum, forall_sum_op);
   forall_reduce_CheckResult("IndexSet::ExecPolicy<omp_parallel_for_segit, simd_exec>",
                             ref_sum, tsum);

#if defined(RAJA_COMPILER_ICC)
   tsum = 0.0;
   forall_sum< IndexSet::ExecPolicy<seq_segit, cilk_for_exec> >(
                                   hindex, &tsum, forall_sum_op);
   forall_reduce_CheckResult("IndexSet::ExecPolicy<seq_segit, cilk_for_exec>",
                             ref_sum, tsum);

   tsum = 0.0;
   forall_sum< IndexSet::ExecPolicy<cilk_for_segit, seq_exec> >(
                                   hindex, &tsum, forall_sum_op);
   forall_reduce_CheckResult("IndexSet::ExecPolicy<cilk_for_segit, seq_exec>",
                             ref_sum, tsum);

   tsum = 0.0;
   forall_sum< IndexSet::ExecPolicy<cilk_for_segit, simd_exec> >(
                                   hindex, &tsum, forall_sum_op);
   forall_reduce_CheckResult("IndexSet::ExecPolicy<cilk_for_segit, simd_exec>",
                             ref_sum, tsum);

   tsum = 0.0;
   forall_sum< IndexSet::ExecPolicy<cilk_for_segit, omp_parallel_for_exec> >(
                                   hindex, &tsum, forall_sum_op);
   forall_reduce_CheckResult("IndexSet::ExecPolicy<cilk_for_segit, omp_parallel_for_exec>",
                             ref_sum, tsum);

   tsum = 0.0;
   forall_sum< IndexSet::ExecPolicy<omp_parallel_for_segit, cilk_for_exec> >(
                                   hindex, &tsum, forall_sum_op);
   forall_reduce_CheckResult("IndexSet::ExecPolicy<omp_parallel_for_segit, cilk_for_exec>",
                             ref_sum, tsum);
#endif

#endif

   std::cout << "\n END RAJA::forall SUM tests...." << std::endl;
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
   IndexSet hindex;
   buildIndexSet( hindex );

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
#if defined(RAJA_USE_STL)
   std::vector<Index_type> is_indices = getIndices(hindex);
#else
   RAJAVec<Index_type> is_indices = getIndices(hindex);
#endif


///////////////////////////////////////////////////////////////////////////
//
// Run RAJA::forall_min loop reduction tests...
//
///////////////////////////////////////////////////////////////////////////

   //
   // Make all array values positive...
   //
   for (Index_type ic = 0; ic < array_length; ++ic) {
      parent[ic] = std::abs( parent[ic] );
   }

   //
   // Generate reference result to check correctness
   //
   const Index_type ref_min_indx =
   static_cast<Index_type>(is_indices[is_indices.size()/2]);
   const Real_type  ref_min_val  = -100.0;

   parent[ref_min_indx] = ref_min_val;

#if 1
   std::cout << "ref_min_indx = " << ref_min_indx << std::endl;
   std::cout << "ref_min_val = " << ref_min_val << std::endl;
   std::cout << "parent[ref_min_indx] = " << parent[ref_min_indx] << std::endl;
#endif

   ntests_run = 0; 
   ntests_passed = 0; 

   runForAllMinTests( hindex, parent, ref_min_val );

   std::cout << "\n forall_min() : " << ntests_passed << " / " << ntests_run << std::endl; 

///////////////////////////////////////////////////////////////////////////
//
// Run RAJA::forall_sum loop reduction tests...
//
///////////////////////////////////////////////////////////////////////////

   //
   // Generate reference result to check correctness
   //
   Real_type  ref_sum  = 0.0;

   for (Index_type i = 0; i < is_indices.size(); ++i) {
      ref_sum += parent[ is_indices[i] ];
   }

#if 0
   std::cout << "ref_sum = " << ref_sum << std::endl;
#endif

   ntests_run = 0; 
   ntests_passed = 0; 

   runForAllSumTests( hindex, parent, ref_sum );

   std::cout << "\n\n forall_sum() : " << ntests_passed << " / " << ntests_run << std::endl; 


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

