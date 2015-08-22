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
// Run RAJA min reduce tests with available RAJA execution policies....
//
///////////////////////////////////////////////////////////////////////////
void runMinReduceTests( const IndexSet& hindex, 
                        Real_ptr parent, 
                        Real_type ref_min )
{
// Run various min reductions and check results against reference...

   std::cout << "\n\n   BEGIN RAJA::forall MIN REDUCE tests...." << std::endl;

{
   std::string policy("ExecPolicy<seq_segit, seq_exec>");
   std::cout << "\n" << policy << " tests...\n";

   ReduceMin<seq_reduce, Real_type> tmin0(1.0e+20);
   ReduceMin<seq_reduce, Real_type> tmin1(-200.0);

   int loops = 2;

   for (int k = 1; k <= loops; ++ k) {

//    std::cout << "k = " << k << std::endl;

      forall< IndexSet::ExecPolicy<seq_segit, seq_exec> >( hindex,
         [=] (Index_type idx) {
         tmin0.min(k*parent[idx]);
         tmin1.min(parent[idx]);
      } );

      forall_reduce_CheckResult("ReduceMin:" + policy + ": tmin0",
                                k*ref_min, tmin0);
      forall_reduce_CheckResult("ReduceMin:" + policy + ": tmin1",
                                -200.0, tmin1);
   }
}

{ 
   std::string policy("ExecPolicy<seq_segit, omp_parallel_for_exec>");
   std::cout << "\n" << policy << " tests...\n";

   ReduceMin<omp_reduce, Real_type> tmin0(1.0e+20);
   ReduceMin<omp_reduce, Real_type> tmin1(-200.0);

   int loops = 2;

   for (int k = 1; k <= loops; ++ k) {

//    std::cout << "k = " << k << std::endl;

      forall< IndexSet::ExecPolicy<seq_segit, omp_parallel_for_exec> >( hindex,
         [=] (Index_type idx) {
         tmin0.min(k*parent[idx]);
         tmin1.min(parent[idx]);
      } );

      forall_reduce_CheckResult("ReduceMin:" + policy + ": tmin0",
                                k*ref_min, tmin0);
      forall_reduce_CheckResult("ReduceMin:" + policy + ": tmin1",
                                -200.0, tmin1);
   }
}

{
   std::string policy("ExecPolicy<omp_parallel_for_segit, seq_exec>");
   std::cout << "\n" << policy << " tests...\n";

   ReduceMin<omp_reduce, Real_type> tmin0(1.0e+20);
   ReduceMin<omp_reduce, Real_type> tmin1(-200.0);

   int loops = 2;

   for (int k = 1; k <= loops; ++ k) {

//    std::cout << "k = " << k << std::endl;

      forall< IndexSet::ExecPolicy<omp_parallel_for_segit, seq_exec> >( hindex,
         [=] (Index_type idx) {
         tmin0.min(k*parent[idx]);
         tmin1.min(parent[idx]);
      } );

      forall_reduce_CheckResult("ReduceMin:" + policy + ": tmin0",
                                k*ref_min, tmin0);
      forall_reduce_CheckResult("ReduceMin:" + policy + ": tmin1",
                                -200.0, tmin1);
   }
}

   std::cout << "\n   END RAJA::forall MIN REDUCE tests... " << std::endl;
}


///////////////////////////////////////////////////////////////////////////
//
// Run RAJA max reduce tests with available RAJA execution policies....
//
///////////////////////////////////////////////////////////////////////////
void runMaxReduceTests( const IndexSet& hindex, 
                        Real_ptr parent, 
                        Real_type ref_max )
{
// Run various max reductions and check results against reference...

   std::cout << "\n\n   BEGIN RAJA::forall MAX REDUCE tests...." << std::endl;

{
   std::string policy("ExecPolicy<seq_segit, seq_exec>");
   std::cout << "\n" << policy << " tests...\n";

   ReduceMax<seq_reduce, Real_type> tmax0(-1.0e+20);
   ReduceMax<seq_reduce, Real_type> tmax1(200.0);

   int loops = 2;

   for (int k = 1; k <= loops; ++ k) {

//    std::cout << "k = " << k << std::endl;

      forall< IndexSet::ExecPolicy<seq_segit, seq_exec> >( hindex,
         [=] (Index_type idx) {
         tmax0.max(k*parent[idx]);
         tmax1.max(parent[idx]);
      } );


      forall_reduce_CheckResult("ReduceMax:" + policy + ": tmax0",
                                k*ref_max, tmax0);
      forall_reduce_CheckResult("ReduceMax:" + policy + ": tmax1",
                                200.0, tmax1);
   }
}

{ 
   std::string policy("ExecPolicy<seq_segit, omp_parallel_for_exec>");
   std::cout << "\n" << policy << " tests...\n";

   ReduceMax<omp_reduce, Real_type> tmax0(-1.0e+20);
   ReduceMax<omp_reduce, Real_type> tmax1(200.0);

   int loops = 2;

   for (int k = 1; k <= loops; ++ k) {

//    std::cout << "k = " << k << std::endl;

      forall< IndexSet::ExecPolicy<seq_segit, omp_parallel_for_exec> >( hindex,
         [=] (Index_type idx) {
         tmax0.max(k*parent[idx]);
         tmax1.max(parent[idx]);
      } );

      forall_reduce_CheckResult("ReduceMax:" + policy + ": tmax0",
                                k*ref_max, tmax0);
      forall_reduce_CheckResult("ReduceMax:" + policy + ": tmax1",
                                200.0, tmax1);
   }
}

{
   std::string policy("ExecPolicy<omp_parallel_for_segit, seq_exec>");
   std::cout << "\n" << policy << " tests...\n";

   ReduceMax<omp_reduce, Real_type> tmax0(-1.0e+20);
   ReduceMax<omp_reduce, Real_type> tmax1(200.0);

   int loops = 2;

   for (int k = 1; k <= loops; ++ k) {

//    std::cout << "k = " << k << std::endl;

      forall< IndexSet::ExecPolicy<omp_parallel_for_segit, seq_exec> >( hindex,
         [=] (Index_type idx) {
         tmax0.max(k*parent[idx]);
         tmax1.max(parent[idx]);
      } );

      forall_reduce_CheckResult("ReduceMax:" + policy + ": tmax0",
                                k*ref_max, tmax0);
      forall_reduce_CheckResult("ReduceMax:" + policy + ": tmax1",
                                200.0, tmax1);
   }
}

   std::cout << "\n   END RAJA::forall MAX REDUCE tests... " << std::endl;
}


///////////////////////////////////////////////////////////////////////////
//
// Run RAJA sum reduce tests with available RAJA execution policies....
//
///////////////////////////////////////////////////////////////////////////
void runSumReduceTests( const IndexSet& hindex, 
                        Real_ptr parent, 
                        Real_type ref_sum )
{
// Run various sum reductions and check results against reference...

   std::cout << "\n\n   BEGIN RAJA::forall SUM REDUCE tests...." << std::endl;

{
   std::string policy("ExecPolicy<seq_segit, seq_exec>");
   std::cout << "\n" << policy << " tests...\n";

   ReduceSum<seq_reduce, Real_type> tsum0(0.0);
   ReduceSum<seq_reduce, Real_type> tsum1(5.0);

   int loops = 2;

   for (int k = 1; k <= loops; ++k) {

//    std::cout << "k = " << k << std::endl;

      forall< IndexSet::ExecPolicy<seq_segit, seq_exec> >( hindex,
         [=] (Index_type idx) {
         tsum0 += parent[idx];
         tsum1 += 1.0;
      } );

      forall_reduce_CheckResult("ReduceSum:" + policy + ": tsum0",
                                k*ref_sum, tsum0);
      forall_reduce_CheckResult("ReduceSum:" + policy + ": tsum1",
                                k*hindex.getLength() + 5.0, tsum1);
   }
}

{
   std::string policy("ExecPolicy<seq_segit, omp_parallel_for_exec>");
   std::cout << "\n" << policy << " tests...\n";

   ReduceSum<omp_reduce, Real_type> tsum0(0.0);
   ReduceSum<omp_reduce, Real_type> tsum1(5.0);

   int loops = 2;

   for (int k = 1; k <= loops; ++k) {

//    std::cout << "k = " << k << std::endl;

      forall< IndexSet::ExecPolicy<seq_segit, omp_parallel_for_exec> >( hindex,
         [=] (Index_type idx) {
         tsum0 += parent[idx];
         tsum1 += 1.0;
      } );

      forall_reduce_CheckResult("ReduceSum:" + policy + ": tsum0",
                                k*ref_sum, tsum0);
      forall_reduce_CheckResult("ReduceSum:" + policy + ": tsum1",
                                k*hindex.getLength() + 5.0, tsum1);
   }
}

{
   std::string policy("ExecPolicy<omp_parallel_for_segit, seq_exec>");
   std::cout << "\n" << policy << " tests...\n";

   ReduceSum<omp_reduce, Real_type> tsum0(0.0);
   ReduceSum<omp_reduce, Real_type> tsum1(5.0);

   int loops = 2;

   for (int k = 1; k <= loops; ++k) {

//    std::cout << "k = " << k << std::endl;

      forall< IndexSet::ExecPolicy<omp_parallel_for_segit, seq_exec> >( hindex,
         [=] (Index_type idx) {
         tsum0 += parent[idx];
         tsum1 += 1.0;
      } );

      forall_reduce_CheckResult("ReduceSum:" + policy + ": tsum0",
                                k*ref_sum, tsum0);
      forall_reduce_CheckResult("ReduceSum:" + policy + ": tsum1",
                                k*hindex.getLength() + 5.0, tsum1);
   }
}


   std::cout << "\n END   RAJA::forall SUM REDUCE tests...." << std::endl;
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
// Run RAJA min reduction tests...
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

   runMinReduceTests( hindex, parent, ref_min_val );

   std::cout << "\n MIN reduction : " << ntests_passed << " / " << ntests_run << std::endl; 


///////////////////////////////////////////////////////////////////////////
//
// Run RAJA max reduction tests...
//
///////////////////////////////////////////////////////////////////////////

   //
   // Make all array values negative...
   //
   for (Index_type ic = 0; ic < array_length; ++ic) {
      parent[ic] = -std::abs( parent[ic] );
   }

   //
   // Generate reference result to check correctness
   //
   const Index_type ref_max_indx =
   static_cast<Index_type>(is_indices[is_indices.size()/2]);
   const Real_type  ref_max_val  = 100.0;

   parent[ref_max_indx] = ref_max_val;

#if 0
   std::cout << "ref_max_indx = " << ref_max_indx << std::endl;
   std::cout << "ref_max_val = " << ref_max_val << std::endl;
   std::cout << "parent[ref_max_indx] = " << parent[ref_max_indx] << std::endl;
#endif

   ntests_run = 0;
   ntests_passed = 0;

   runMaxReduceTests( hindex, parent, ref_max_val );

   std::cout << "\n MAX reduction : " << ntests_passed << " / " << ntests_run <<
 std::endl;


///////////////////////////////////////////////////////////////////////////
//
// Run RAJA sum reduction tests...
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

   runSumReduceTests( hindex, parent, ref_sum );

   std::cout << "\n\n SUM reduction : " << ntests_passed << " / " << ntests_run << std::endl; 


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

