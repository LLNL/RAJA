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

#include "RAJA/RAJA.hpp"
#include "RAJA/util/Timer.hpp"

#include <iostream>
#include "memoryManager.hpp"

/*
 *  Batched Matrix Multiply Example
 *
 *  Carries out matrix multiplication
 *  for NMAT matrices. Here we assume
 *  matrices are small (3 x 3).
 *  Each iteration of the RAJA forall loop
 *  will multiply a matrix. 
 *
 *  Additionally, we explore performance for two data layouts. 
 *  Layout 1: Assumes matrices are contiguous in memory
 *  i.e. [a00, a01, a02, ... ]
 *
 * Layout 2: Multiplies matrices assuming matries entries are grouped together
 * i.e. [a00, b00, c00, a01, b01, c01 ... ]
 *
 *  RAJA features shown:
 *    -  RAJA View
 *    -  RAJA make_permuted_layout
 *
 * If CUDA is enabled, CUDA unified memory is used. 
 */

using RAJA::Index_type;

/*
  CUDA_BLOCK_SIZE - specifies the number of threads in a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 256;
using GPUPol = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;
#endif
using CPUPol = RAJA::loop_exec;



//Dimensions of matrices
const Index_type NCOLS = 3;
const Index_type NROWS = 3;
const Index_type NMAT  = 80000000;
const Index_type NELEM = NCOLS*NROWS*NMAT;

//Number of iterations
const int NITER = 21;

//
// Functions for comparing outputs
//
template<typename T, typename U>
void compareOutput(T C, U Cl2,Index_type N);

//
template<typename Pol, typename T>
void multiplyViewCPU(const T RAJA_RESTRICT Aview, const T RAJA_RESTRICT Bview,
                     T const RAJA_RESTRICT Cview, const Index_type N);


template<typename Pol, typename T>
void multiplyViewGPU(const T RAJA_RESTRICT Aview, const T RAJA_RESTRICT Bview,
                     T const RAJA_RESTRICT Cview, const Index_type N);


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA batched matrix multiplication example...\n";

  double myMin;
  srand(time(NULL));
  auto timer = RAJA::Timer();

//
//Space for data in layout 1
//
  double * A = memoryManager::allocate<double>(NCOLS*NROWS*NMAT);
  double * B = memoryManager::allocate<double>(NCOLS*NROWS*NMAT);
  double * C = memoryManager::allocate<double>(NCOLS*NROWS*NMAT);

//
//Default layout - equivalent to indexing via 
//A[c + NCOLS*(r + NROWS*e)]
//
  auto layout = RAJA::make_permuted_layout({{NMAT, NROWS, NCOLS}}, RAJA::as_array<RAJA::Perm<0,1,2> >::get() );
  RAJA::View<double, RAJA::Layout<3, Index_type, 2> > Aview(A,layout);
  RAJA::View<double, RAJA::Layout<3, Index_type, 2> > Bview(B,layout);
  RAJA::View<double, RAJA::Layout<3, Index_type, 2> > Cview(C,layout);

//
//Stores data in layout 2
//
  double * Al2 = memoryManager::allocate<double>(NCOLS*NROWS*NMAT);
  double * Bl2 = memoryManager::allocate<double>(NCOLS*NROWS*NMAT);
  double * Cl2 = memoryManager::allocate<double>(NCOLS*NROWS*NMAT);
  
//
//Permuted layout - equivalent to indexing via 
//A[e + NELEM*(r + NROWS*e)]
//
  auto layout2 = RAJA::make_permuted_layout({{NMAT, NROWS, NCOLS}}, RAJA::as_array<RAJA::Perm<1,2,0> >::get() );
  RAJA::View<double, RAJA::Layout<3, Index_type, 0> > Al2view(Al2,layout2);
  RAJA::View<double, RAJA::Layout<3, Index_type, 0> > Bl2view(Bl2,layout2);
  RAJA::View<double, RAJA::Layout<3, Index_type, 0> > Cl2view(Cl2,layout2);

//
//Initialize data
//
  for(Index_type e=0; e<NMAT; ++e){
    for(Index_type row=0; row<NROWS; ++row){
      for(Index_type col=0; col<NCOLS; ++col){
        Aview(e,row,col) = rand() % 50 + 1; 
        Bview(e,row,col) = rand() % 50 + 1; 
        Al2view(e,row,col) = Aview(e,row,col);
        Bl2view(e,row,col) = Bview(e,row,col);
      }
    }                 
  }

//-------------------------------------------
//Matrix multiply with layout 1 on the CPU
//
  myMin = std::numeric_limits<double>::max();  
  for(Index_type i=0; i<NITER; ++i){
    
  timer.start();
  multiplyViewCPU<CPUPol>(Aview,Bview,Cview,NMAT);
  timer.stop();
  
  RAJA::Timer::ElapsedType tMin = timer.elapsed();
  if(tMin < myMin) myMin = tMin;
  }
  std::cout<<"Matrix Multiplication with layout 1 run time : "<<myMin<<" seconds"<<std::endl;
//-------------------------------------------

//-------------------------------------------
//Matrix multiply with layout 2 on the CPU
//
  myMin = std::numeric_limits<double>::max();  
  for(Index_type i=0; i<NITER; ++i){
    
  timer.start();
  multiplyViewCPU<CPUPol>(Al2view,Bl2view,Cl2view,NMAT);
  timer.stop();
  
  RAJA::Timer::ElapsedType tMin = timer.elapsed();
  if(tMin < myMin) myMin = tMin;
  }
  std::cout<<"Matrix Multiplication with layout 2 run time : "<<myMin<<" seconds"<<std::endl;
//---------------------------------------------

//
//Compare output
//
  compareOutput(Cview, Cl2view, NELEM);

#if defined(RAJA_ENABLE_CUDA)
//-------------------------------------------
//Matrix multiply with layout 1 on the GPU
//
  myMin = std::numeric_limits<double>::max();  
  for(Index_type i=0; i<NITER; ++i){
    
  timer.start();
  multiplyViewGPU<GPUPol>(Aview,Bview,Cview,NMAT);
  cudaDeviceSyncronize();
  timer.stop();
  
  RAJA::Timer::ElapsedType tMin = timer.elapsed();
  if(tMin < myMin) myMin = tMin;
  }
  std::cout<<"Matrix Multiplication with layout 1 run time : "<<myMin<<" seconds"<<std::endl;
//-------------------------------------------

//-------------------------------------------
//Matrix multiply with layout 2 on the GPU
//
  myMin = std::numeric_limits<double>::max();  
  for(Index_type i=0; i<NITER; ++i){
    
  timer.start();
  multiplyViewGPU<GPUPol>(Al2view,Bl2view,Cl2view,NMAT);
  cudaDeviceSyncronize();
  timer.stop();
  
  RAJA::Timer::ElapsedType tMin = timer.elapsed();
  if(tMin < myMin) myMin = tMin;
  }
  std::cout<<"Matrix Multiplication with layout 2 run time : "<<myMin<<" seconds"<<std::endl;
//---------------------------------------------

//
//Compare output
//
  compareOutput(Cview, Cl2view, NELEM);  
#endif

//
// Clean up.
//
  memoryManager::deallocate(A);
  memoryManager::deallocate(B);
  memoryManager::deallocate(C);
  memoryManager::deallocate(Cl2);

  std::cout << "\n DONE!...\n";

  return 0;
}

template<typename Pol, typename T>
void multiplyViewCPU(const T RAJA_RESTRICT Aview, const T RAJA_RESTRICT Bview,
                     T const RAJA_RESTRICT Cview, const Index_type N)
{
  
    RAJA::forall<Pol>(RAJA::RangeSegment(0, N), [=] (Index_type i) {

      Cview(i,0,0) = Aview(i,0,0)*Bview(i,0,0) + Aview(i,0,1)*Bview(i,1,0) + Aview(i,0,2)*Bview(i,2,0);
      Cview(i,0,1) = Aview(i,0,0)*Bview(i,0,1) + Aview(i,0,1)*Bview(i,1,1) + Aview(i,0,2)*Bview(i,2,1);
      Cview(i,0,2) = Aview(i,0,0)*Bview(i,0,2) + Aview(i,0,1)*Bview(i,1,2) + Aview(i,0,2)*Bview(i,2,2);

      Cview(i,1,0) = Aview(i,1,0)*Bview(i,0,0) + Aview(i,1,1)*Bview(i,1,0) + Aview(i,1,2)*Bview(i,2,0);
      Cview(i,1,1) = Aview(i,1,0)*Bview(i,0,1) + Aview(i,1,1)*Bview(i,1,1) + Aview(i,1,2)*Bview(i,2,1);
      Cview(i,1,2) = Aview(i,1,0)*Bview(i,0,2) + Aview(i,1,1)*Bview(i,1,2) + Aview(i,1,2)*Bview(i,2,2);

      Cview(i,2,0) = Aview(i,2,0)*Bview(i,0,0) + Aview(i,2,1)*Bview(i,1,0) + Aview(i,2,2)*Bview(i,2,0);
      Cview(i,2,1) = Aview(i,2,0)*Bview(i,0,1) + Aview(i,2,1)*Bview(i,1,1) + Aview(i,2,2)*Bview(i,2,1);
      Cview(i,2,2) = Aview(i,2,0)*Bview(i,0,2) + Aview(i,2,1)*Bview(i,1,2) + Aview(i,2,2)*Bview(i,2,2);
    });

}

#if defined(RAJA_ENABLE_CUDA)
template<typename Pol, typename T>
void multiplyViewGPU(const T RAJA_RESTRICT Aview, const T RAJA_RESTRICT Bview,
                     T const RAJA_RESTRICT Cview, const Index_type N)
{
  
    RAJA::forall<Pol>(RAJA::RangeSegment(0, N), [=] __device__ (Index_type i) {

      Cview(i,0,0) = Aview(i,0,0)*Bview(i,0,0) + Aview(i,0,1)*Bview(i,1,0) + Aview(i,0,2)*Bview(i,2,0);
      Cview(i,0,1) = Aview(i,0,0)*Bview(i,0,1) + Aview(i,0,1)*Bview(i,1,1) + Aview(i,0,2)*Bview(i,2,1);
      Cview(i,0,2) = Aview(i,0,0)*Bview(i,0,2) + Aview(i,0,1)*Bview(i,1,2) + Aview(i,0,2)*Bview(i,2,2);

      Cview(i,1,0) = Aview(i,1,0)*Bview(i,0,0) + Aview(i,1,1)*Bview(i,1,0) + Aview(i,1,2)*Bview(i,2,0);
      Cview(i,1,1) = Aview(i,1,0)*Bview(i,0,1) + Aview(i,1,1)*Bview(i,1,1) + Aview(i,1,2)*Bview(i,2,1);
      Cview(i,1,2) = Aview(i,1,0)*Bview(i,0,2) + Aview(i,1,1)*Bview(i,1,2) + Aview(i,1,2)*Bview(i,2,2);

      Cview(i,2,0) = Aview(i,2,0)*Bview(i,0,0) + Aview(i,2,1)*Bview(i,1,0) + Aview(i,2,2)*Bview(i,2,0);
      Cview(i,2,1) = Aview(i,2,0)*Bview(i,0,1) + Aview(i,2,1)*Bview(i,1,1) + Aview(i,2,2)*Bview(i,2,1);
      Cview(i,2,2) = Aview(i,2,0)*Bview(i,0,2) + Aview(i,2,1)*Bview(i,1,2) + Aview(i,2,2)*Bview(i,2,2);
    });
}
#endif


//
// Compare output
//
template<typename T, typename U>
void compareOutput(T C, U Cl2, Index_type Nelem)
{

  bool status = true;
  for(Index_type e = 0; e<NMAT; ++e){
    for(Index_type row=0; row<NROWS; ++row){
      for(Index_type col=0; col<NCOLS; ++col){
        double terr = std::abs(C(e,row,col) - Cl2(e,row,col));
        if((terr) > 1e-8)
          {
            status = false;
          }
      }
    }
  }

  if(status==false)
    {
      std::cout<<"Matrix Multiply - fail"<<std::endl;
    }else{
    std::cout<<"Matrix Multiply - pass"<<std::endl;
  }

}



