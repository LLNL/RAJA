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

#include "buildHybrid.hxx"

//
// For tested platforms, only xlc compiler fails to support C++ lambda fcns.
//
#if defined(RAJA_RAJA_PLATFORM_BGQ) 

#if defined(RAJA_COMPILER_XLC12) || defined(RAJA_COMPILER_GNU)
#undef USE_LAMBDA
#else
#define USE_LAMBDA
#endif

#else

#define USE_LAMBDA
//#undef USE_LAMBDA
#endif


#if !defined(USE_LAMBDA)  
// 
// Define functor classes for test loop bodies
// 

class forall_TestOp
{
public:
   forall_TestOp(Real_ptr child, Real_ptr parent) :
      m_child(child), m_parent(parent) { ; }

   void operator() (Index_type idx)
   {
      m_child[idx] = m_parent[idx] * m_parent[idx];
   }

   Real_ptr m_child;
   Real_ptr m_parent;
};

class forall_minloc_TestOp
{
public:
   forall_minloc_TestOp(Real_ptr parent) : m_parent(parent) { ; }

   void operator() (Index_type idx, double* myMin, int* myLoc)
   {
      if ( *myMin > m_parent[idx] ) {
         *myMin = m_parent[idx];
         *myLoc = idx;
      } 
   }

   Real_ptr m_parent;
};

class forall_maxloc_TestOp
{
public:
   forall_maxloc_TestOp(Real_ptr parent) : m_parent(parent) { ; }

   void operator() (Index_type idx, double* myMax, int* myLoc)
   {
      if ( *myMax < m_parent[idx] ) {
         *myMax = m_parent[idx];
         *myLoc = idx;
      }
   }

   Real_ptr m_parent;
};

class forall_sum_TestOp
{
public:
   forall_sum_TestOp(Real_ptr parent) : m_parent(parent) { ; }

   void operator() (Index_type idx, double* mySum)
   {
      *mySum += m_parent[idx];
   }

   Real_ptr m_parent;
};
#endif


//
//  Simple utility function to check results of forall traversals
//
void forall_CheckResult(const std::string& name,
                        Real_ptr ref_result,  
                        Real_ptr to_check,  
                        Index_type iset_len)
{
   bool is_correct = true;
   for (Index_type i = 0 ; i < iset_len && is_correct; ++i) {
      is_correct &= ref_result[i] == to_check[i];
   }
   
   std::cout << name << " is " 
             << (is_correct ? "CORRECT" : "WRONG") << std::endl; 
}

//
//  Simple utility function to check results of min/maxloc reductions
//
void forall_reduceloc_CheckResult(const std::string& name,
                                  const Real_type ref_val,
                                  const Index_type ref_idx,
                                  Real_type check_val,
                                  Index_type check_idx)
{
   bool is_correct = (ref_val == check_val) && (ref_idx == check_idx);
  
   std::cout << name << " is ";
   if ( is_correct ) {
      std::cout << "CORRECT" << std::endl;
   } else { 
      std::cout << "WRONG" << std::endl;
      std::cout << "\tref_val = " << ref_val << std::endl; 
      std::cout << "\tref_idx = " << ref_idx << std::endl; 
      std::cout << "\tcheck_val = " << check_val << std::endl; 
      std::cout << "\tcheck_idx = " << check_idx << std::endl; 
   }
}

//
//  Simple utility function to check results of min/maxloc reductions
//
void forall_reduce_CheckResult(const std::string& name,
                               const Real_type ref_val,
                               Real_type check_val)
{
   bool is_correct = (ref_val == check_val);
 
   std::cout << name << " is ";
   if ( is_correct ) {
      std::cout << "CORRECT" << std::endl;
   } else { 
      std::cout << "WRONG" << std::endl;
      std::cout << "\tref_val = " << ref_val << std::endl; 
      std::cout << "\tcheck_val = " << check_val << std::endl; 
   }
}



int main(int argc, char *argv[])
{

// 
//  Testing different methods to construct hybrid segments.
//  They should generate equivalent results.
//
#if 0
   HybridISet* hindex_build = buildHybrid_addIndices();
   HybridISet& hindex = *hindex_build;
#else
   bool use_vector = true;
   HybridISet* hindex_build = buildHybrid_addSegments(use_vector);
   HybridISet& hindex = *hindex_build;
#endif

#if 0
   std::cout << std::endl << std::endl;
   hindex.print(std::cout);
   std::cout << std::endl << std::endl;
#endif

   //
   // Compute max index in hybrid index set.
   //
   Index_type max_hindex = 0;
   forall< std::pair<seq_segit, seq_exec> >(hindex, [&] (Index_type idx) {
      max_hindex = std::max( max_hindex, idx );
   } );
   

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
   std::vector<Index_type> is_indices = getIndices(hindex);


///////////////////////////////////////////////////////////////////////////
//
// Check correctness of RAJA forall iteration templates with 
// available RAJA execution policies....
//
///////////////////////////////////////////////////////////////////////////

   //
   // Generate reference result to check correctness
   //
   for (Index_type i = 0; i < is_indices.size(); ++i) {
      child_ref[ is_indices[i] ] = 
         parent[ is_indices[i] ] * parent[ is_indices[i] ];
   }

//
// Loop body for forall tests.
//   
#if defined(USE_LAMBDA)
   auto forall_op = [&] (Index_type idx) {
                       child[idx] = parent[idx] * parent[idx]; };
#else
   auto forall_op = forall_TestOp(child, parent);
#endif
   
// Run various traversals and check results against reference...
   std::cout << "\n\n BEGIN forall tests...\n" << std::endl;

   forall< std::pair<seq_segit, seq_exec> >(hindex, forall_op);
   forall_CheckResult("pair<seq_segit, seq_exec>", 
                      child_ref, child, hindex.getLength());

   forall< std::pair<seq_segit, simd_exec> >(hindex, forall_op);
   forall_CheckResult("pair<seq_segit, simd_exec>", 
                      child_ref, child, hindex.getLength());

   forall< std::pair<seq_segit, omp_parallel_for_exec> >(hindex, forall_op);
   forall_CheckResult("pair<seq_segit, omp_parallel_for_exec>", 
                      child_ref, child, hindex.getLength());

   forall< std::pair<omp_parallel_for_segit, seq_exec> >(hindex, forall_op);
   forall_CheckResult("pair<omp_parallel_for_segit, seq_exec>", 
                      child_ref, child, hindex.getLength());

   forall< std::pair<omp_parallel_for_segit, simd_exec> >(hindex, forall_op);
   forall_CheckResult("pair<omp_parallel_for_segit, simd_exec>", 
                      child_ref, child, hindex.getLength());

#if defined(RAJA_COMPILER_ICC)
   forall< std::pair<seq_segit, cilk_for_exec> >(hindex, forall_op);
   forall_CheckResult("pair<seq_segit, cilk_for_exec>", 
                      child_ref, child, hindex.getLength());

   forall< std::pair<cilk_for_segit, seq_exec> >(hindex, forall_op);
   forall_CheckResult("pair<cilk_for_segit, seq_exec>", 
                      child_ref, child, hindex.getLength());

   forall< std::pair<cilk_for_segit, simd_exec> >(hindex, forall_op);
   forall_CheckResult("pair<cilk_for_segit, simd_exec>", 
                      child_ref, child, hindex.getLength());

   forall< std::pair<cilk_for_segit, omp_parallel_for_exec> >(hindex, forall_op);
   forall_CheckResult("pair<cilk_for_segit, omp_parallel_for_exec>", 
                      child_ref, child, hindex.getLength());

   forall< std::pair<omp_parallel_for_segit, cilk_for_exec> >(hindex, forall_op);
   forall_CheckResult("pair<omp_parallel_for_segit, cilk_for_exec>", 
                      child_ref, child, hindex.getLength());
#endif

   std::cout << "\n END forall tests...\n\n" << std::endl;

#if 0 // print output for manual checking...
   std::cout << "\n CHILD ARRAY OUTPUT... " << std::endl;

   for (Index_type ic = 0; ic < array_length; ++ic) {
      std::cout << "child[" << ic << "] = " << child[ic] << std::endl; ;
   }
#endif


///////////////////////////////////////////////////////////////////////////
//
// Check correctness of RAJA minloc reduction templates with 
// available RAJA execution policies....
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

#if 0
   std::cout << "ref_min_indx = " << ref_min_indx << std::endl;
   std::cout << "ref_min_val = " << ref_min_val << std::endl;
   std::cout << "parent[ref_min_indx] = " << parent[ref_min_indx] << std::endl;
#endif

//
// Loop body for minloc reductions. 
//   
#if defined(USE_LAMBDA)
   auto forall_minloc_op = [&] (Index_type idx, double* myMin, int* myLoc) 
                               { 
                                  if ( *myMin > parent[idx] ) {
                                     *myMin = parent[idx];
                                     *myLoc = idx;
                                  }
                                };
#else
   auto forall_minloc_op = forall_minloc_TestOp(parent);
#endif

// Run various minloc traversals and check results against reference...
   Real_type tmin = 1.0e+20 ;
   Index_type tloc = -1 ;

   std::cout << "\n\n BEGIN forall_minloc tests...\n" << std::endl;

   forall_minloc< std::pair<seq_segit, seq_exec> >(
                                      hindex, &tmin, &tloc, forall_minloc_op);
   forall_reduceloc_CheckResult("pair<seq_segit, seq_exec>",
                                ref_min_val, ref_min_indx, tmin, tloc);

   tmin = 1.0e+20 ;
   tloc = -1 ;
   forall_minloc< std::pair<seq_segit, simd_exec> >(
                                      hindex, &tmin, &tloc, forall_minloc_op);
   forall_reduceloc_CheckResult("pair<seq_segit, simd_exec>",
                                ref_min_val, ref_min_indx, tmin, tloc);

   tmin = 1.0e+20 ;
   tloc = -1 ;
   forall_minloc< std::pair<seq_segit, omp_parallel_for_exec> >(
                                      hindex, &tmin, &tloc, forall_minloc_op);
   forall_reduceloc_CheckResult("pair<seq_segit, omp_parallel_for_exec>",
                                ref_min_val, ref_min_indx, tmin, tloc);

   tmin = 1.0e+20 ;
   tloc = -1 ;
   forall_minloc< std::pair<omp_parallel_for_segit, seq_exec> >(
                                      hindex, &tmin, &tloc, forall_minloc_op);
   forall_reduceloc_CheckResult("pair<omp_parallel_for_segit, seq_exec>",
                                ref_min_val, ref_min_indx, tmin, tloc);

   tmin = 1.0e+20 ;
   tloc = -1 ;
   forall_minloc< std::pair<omp_parallel_for_segit, simd_exec> >(
                                      hindex, &tmin, &tloc, forall_minloc_op);
   forall_reduceloc_CheckResult("pair<omp_parallel_for_segit, simd_exec>",
                                ref_min_val, ref_min_indx, tmin, tloc);

#if defined(RAJA_COMPILER_ICC)
   tmin = 1.0e+20 ;
   tloc = -1 ;
   forall_minloc< std::pair<seq_segit, cilk_for_exec> >(
                                      hindex, &tmin, &tloc, forall_minloc_op);
   forall_reduceloc_CheckResult("pair<seq_segit, cilk_for_exec>",
                                ref_min_val, ref_min_indx, tmin, tloc);

   tmin = 1.0e+20 ;
   tloc = -1 ;
   forall_minloc< std::pair<cilk_for_segit, seq_exec> >(
                                      hindex, &tmin, &tloc, forall_minloc_op);
   forall_reduceloc_CheckResult("pair<cilk_for_segit, seq_exec>",
                                ref_min_val, ref_min_indx, tmin, tloc);

   tmin = 1.0e+20 ;
   tloc = -1 ;
   forall_minloc< std::pair<cilk_for_segit, simd_exec> >(
                                      hindex, &tmin, &tloc, forall_minloc_op);
   forall_reduceloc_CheckResult("pair<cilk_for_segit, simd_exec>",
                                ref_min_val, ref_min_indx, tmin, tloc);

   tmin = 1.0e+20 ;
   tloc = -1 ;
   forall_minloc< std::pair<cilk_for_segit, omp_parallel_for_exec> >(
                                      hindex, &tmin, &tloc, forall_minloc_op);
   forall_reduceloc_CheckResult("pair<cilk_for_segit, omp_parallel_for_exec>",
                                ref_min_val, ref_min_indx, tmin, tloc);

   tmin = 1.0e+20 ;
   tloc = -1 ;
   forall_minloc< std::pair<omp_parallel_for_segit, cilk_for_exec> >(
                                      hindex, &tmin, &tloc, forall_minloc_op);
   forall_reduceloc_CheckResult("pair<omp_parallel_for_segit, cilk_for_exec>",
                                ref_min_val, ref_min_indx, tmin, tloc);
#endif

   std::cout << "\n END forall_minloc tests...\n\n" << std::endl;


///////////////////////////////////////////////////////////////////////////
//
// Check correctness of RAJA maxloc reduction templates with
// available RAJA execution policies....
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

//
// Loop body for maxloc reductions. 
//   
#if defined(USE_LAMBDA)
   auto forall_maxloc_op = [&] (Index_type idx, double* myMax, int* myLoc) 
                               {
                                  if ( *myMax < parent[idx] ) {
                                     *myMax = parent[idx];
                                     *myLoc = idx;
                                  }
                                };
#else
   auto forall_maxloc_op = forall_maxloc_TestOp(parent);
#endif

// Run various maxloc traversals and check results against reference...
   Real_type tmax = -1.0e+20 ;
   tloc = -1 ;

   std::cout << "\n\n BEGIN forall_maxloc tests...\n" << std::endl;

   forall_maxloc< std::pair<seq_segit, seq_exec> >(
                                      hindex, &tmax, &tloc, forall_maxloc_op);
   forall_reduceloc_CheckResult("pair<seq_segit, seq_exec>",
                                ref_max_val, ref_max_indx, tmax, tloc);

   tmax = -1.0e+20 ;
   tloc = -1 ;
   forall_maxloc< std::pair<seq_segit, simd_exec> >(
                                      hindex, &tmax, &tloc, forall_maxloc_op);
   forall_reduceloc_CheckResult("pair<seq_segit, simd_exec>",
                                ref_max_val, ref_max_indx, tmax, tloc);

   tmax = -1.0e+20 ;
   tloc = -1 ;
   forall_maxloc< std::pair<seq_segit, omp_parallel_for_exec> >(
                                      hindex, &tmax, &tloc, forall_maxloc_op);
   forall_reduceloc_CheckResult("pair<seq_segit, omp_parallel_for_exec>",
                                ref_max_val, ref_max_indx, tmax, tloc);

   tmax = -1.0e+20 ;
   tloc = -1 ;
   forall_maxloc< std::pair<omp_parallel_for_segit, seq_exec> >(
                                      hindex, &tmax, &tloc, forall_maxloc_op);
   forall_reduceloc_CheckResult("pair<omp_parallel_for_segit, seq_exec>",
                                ref_max_val, ref_max_indx, tmax, tloc);

   tmax = -1.0e+20 ;
   tloc = -1 ;
   forall_maxloc< std::pair<omp_parallel_for_segit, simd_exec> >(
                                      hindex, &tmax, &tloc, forall_maxloc_op);
   forall_reduceloc_CheckResult("pair<omp_parallel_for_segit, simd_exec>",
                                ref_max_val, ref_max_indx, tmax, tloc);

#if defined(RAJA_COMPILER_ICC) 
   tmax = -1.0e+20 ;
   tloc = -1 ;
   forall_maxloc< std::pair<seq_segit, cilk_for_exec> >(
                                      hindex, &tmax, &tloc, forall_maxloc_op);
   forall_reduceloc_CheckResult("pair<seq_segit, cilk_for_exec>",
                                ref_max_val, ref_max_indx, tmax, tloc);

   tmax = -1.0e+20 ;
   tloc = -1 ;
   forall_maxloc< std::pair<cilk_for_segit, seq_exec> >(
                                      hindex, &tmax, &tloc, forall_maxloc_op);
   forall_reduceloc_CheckResult("pair<cilk_for_segit, seq_exec>",
                                ref_max_val, ref_max_indx, tmax, tloc);

   tmax = -1.0e+20 ;
   tloc = -1 ;
   forall_maxloc< std::pair<cilk_for_segit, simd_exec> >(
                                      hindex, &tmax, &tloc, forall_maxloc_op);
   forall_reduceloc_CheckResult("pair<cilk_for_segit, simd_exec>",
                                ref_max_val, ref_max_indx, tmax, tloc);

   tmax = -1.0e+20 ;
   tloc = -1 ;
   forall_maxloc< std::pair<cilk_for_segit, omp_parallel_for_exec> >(
                                      hindex, &tmax, &tloc, forall_maxloc_op);
   forall_reduceloc_CheckResult("pair<cilk_for_segit, omp_parallel_for_exec>",
                                ref_max_val, ref_max_indx, tmax, tloc);

   tmax = -1.0e+20 ;
   tloc = -1 ;
   forall_maxloc< std::pair<omp_parallel_for_segit, cilk_for_exec> >(
                                      hindex, &tmax, &tloc, forall_maxloc_op);
   forall_reduceloc_CheckResult("pair<omp_parallel_for_segit, cilk_for_exec>",
                                ref_max_val, ref_max_indx, tmax, tloc);
#endif

   std::cout << "\n END forall_maxloc tests...\n\n" << std::endl;

///////////////////////////////////////////////////////////////////////////
//
// Check correctness of RAJA sum reduction templates with
// available RAJA execution policies....
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

#if defined(USE_LAMBDA)
   auto forall_sum_op = [&] (Index_type idx, double* mySum) 
                            { 
                               *mySum += parent[idx]; 
                            };
#else
   auto forall_sum_op = forall_sum_TestOp(parent);
#endif

// Run various maxloc traversals and check results against reference...

   Real_type tsum = 0.0;

   std::cout << "\n\n BEGIN forall_sum tests...\n" << std::endl;

   forall_sum< std::pair<seq_segit, seq_exec> >(
                                   hindex, &tsum, forall_sum_op);
   forall_reduce_CheckResult("pair<seq_segit, seq_exec>",
                             ref_sum, tsum);

   tsum = 0.0;
   forall_sum< std::pair<seq_segit, simd_exec> >(
                                   hindex, &tsum, forall_sum_op);
   forall_reduce_CheckResult("pair<seq_segit, simd_exec>",
                             ref_sum, tsum);

   tsum = 0.0;
   forall_sum< std::pair<seq_segit, omp_parallel_for_exec> >(
                                   hindex, &tsum, forall_sum_op);
   forall_reduce_CheckResult("pair<seq_segit, omp_parallel_for_exec>",
                             ref_sum, tsum);

   tsum = 0.0;
   forall_sum< std::pair<omp_parallel_for_segit, seq_exec> >(
                                   hindex, &tsum, forall_sum_op);
   forall_reduce_CheckResult("pair<omp_parallel_for_segit, seq_exec>",
                             ref_sum, tsum);

   tsum = 0.0;
   forall_sum< std::pair<omp_parallel_for_segit, simd_exec> >(
                                   hindex, &tsum, forall_sum_op);
   forall_reduce_CheckResult("pair<omp_parallel_for_segit, simd_exec>",
                             ref_sum, tsum);

#if defined(RAJA_COMPILER_ICC)
   tsum = 0.0;
   forall_sum< std::pair<seq_segit, cilk_for_exec> >(
                                   hindex, &tsum, forall_sum_op);
   forall_reduce_CheckResult("pair<seq_segit, cilk_for_exec>",
                             ref_sum, tsum);

   tsum = 0.0;
   forall_sum< std::pair<cilk_for_segit, seq_exec> >(
                                   hindex, &tsum, forall_sum_op);
   forall_reduce_CheckResult("pair<cilk_for_segit, seq_exec>",
                             ref_sum, tsum);

   tsum = 0.0;
   forall_sum< std::pair<cilk_for_segit, simd_exec> >(
                                   hindex, &tsum, forall_sum_op);
   forall_reduce_CheckResult("pair<cilk_for_segit, simd_exec>",
                             ref_sum, tsum);

   tsum = 0.0;
   forall_sum< std::pair<cilk_for_segit, omp_parallel_for_exec> >(
                                   hindex, &tsum, forall_sum_op);
   forall_reduce_CheckResult("pair<cilk_for_segit, omp_parallel_for_exec>",
                             ref_sum, tsum);

   tsum = 0.0;
   forall_sum< std::pair<omp_parallel_for_segit, cilk_for_exec> >(
                                   hindex, &tsum, forall_sum_op);
   forall_reduce_CheckResult("pair<omp_parallel_for_segit, cilk_for_exec>",
                             ref_sum, tsum);
#endif


   std::cout << "\n END forall_sum tests...\n\n" << std::endl;


#if 1 
///////////////////////////////////////////////////////////////////////////
//
// Check some basic conditional HybridISet construction operations....
//
///////////////////////////////////////////////////////////////////////////

#if !defined(RAJA_COMPILER_XLC12)

   std::vector<Index_type> even_indices = 
      getIndicesConditional( hindex, [](Index_type idx) { return !(idx%2);} );

   HybridISet* test_hiset = buildHybridISet(even_indices);

   std::cout << "\n\n INDEX SET WITH EVEN INDICES ONLY..." << std::endl;
   test_hiset->print(std::cout);

   delete test_hiset;

   std::vector<Index_type> less_than_300_indices = 
      getIndicesConditional( hindex, [](Index_type idx) { return (idx<300);} );

   test_hiset = buildHybridISet(less_than_300_indices);

   std::cout << "\n\n INDEX SET WITH INDICES < 300 ONLY..." << std::endl;
   test_hiset->print(std::cout);

   delete test_hiset;

#endif

#endif


//
// Clean up....
//
   free(parent);
   free(child);
   free(child_ref);

   delete hindex_build;

   std::cout << "\n DONE!!! " << std::endl;

   return 0 ;
}

