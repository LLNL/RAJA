//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for atomic operations
///

#include <RAJA/RAJA.hpp>
#include "RAJA_gtest.hpp"


template <typename RegionPolicy>
void testRegion()
{
    
  int N = 100;
  int *A = new int[N];

  for(int i=0; i<N; ++i){
    A[i] = 0; 
  }
  
  RAJA::Region<RegionPolicy>([=] (){

      RAJA::forall<RAJA::omp_for_exec>(RAJA::RangeSegment(0,N), [=](int i){
          A[i] += 1; 
        });    
      
      
      RAJA::forall<RAJA::omp_for_exec>(RAJA::RangeSegment(0,N), [=](int i){
          A[i] += 1; 
        });      
      
    });
  
  for(int i=0; i<N; ++i){
    EXPECT_EQ(A[i],2);
  }

}

template<typename ExecPol>
void testRegionPol()
{

  testRegion<ExecPol>();
}

TEST(Region, basic_Functions){

  testRegionPol<RAJA::seq_region>();
  
#ifdef RAJA_ENABLE_OPENMP
  testRegionPol<RAJA::omp_region>();
#endif  
}
