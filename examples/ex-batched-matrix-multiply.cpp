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

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <chrono>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Batched Matrix Multiply Example
 *
 *  Multiplies NMAT NCOLSxNROWS matrices.
 *  Illustrates how RAJA views allow developers
 *  to easily change between data layouts. 
 *  Notibly we demonstrate how tailoring data layouts
 *  to specific compute architures enhances performances.
 *
 *  RAJA features shown:
 *    -  RAJA View
 *    -  RAJA make_permuted_layout
 *
 * If CUDA is enabled, CUDA unified memory is used. 
 */

using RAJA::Index_type;
using Clock = std::chrono::high_resolution_clock;

#define A(elem, row, col) A[col + NCOLS*(row + elem*NROWS)]
#define B(elem, row, col) B[col + NCOLS*(row + elem*NROWS)]
#define C1(elem, row, col) C1[col + NCOLS*(row + elem*NROWS)]

#define Al2(elem, row, col) Al2[elem + NMAT*(col + row*NCOLS)]
#define Bl2(elem, row, col) Bl2[elem + NMAT*(col + row*NCOLS)]
#define C2(elem, row, col) C2[elem + NMAT*(col + row*NCOLS)]


/*
  CUDA_BLOCK_SIZE - specifies the number of threads in a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 256;
using KernelPol = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;
#else
using KernelPol = RAJA::simd_exec;
#endif

//Dimensions of matrices
const Index_type NCOLS = 3;
const Index_type NROWS = 3;
const Index_type NMAT  = 8000000;
const Index_type NELEM = NCOLS*NROWS*NMAT;

//Number of iterations
const int NITER = 23;

//
// Functions for comparing outputs
//
void compareOutput(double *C, double *CComp,Index_type N);

//
//Version 1:
//Multiplies matrices assuming entries in a matrix are grouped together
//i.e. [a00, a01, a02, ... ]
//
template<typename myPolicy>
void multiplyVer1(const double * const RAJA_RESTRICT A, const double * const RAJA_RESTRICT B,
                  double * const RAJA_RESTRICT C, const Index_type N);
//
//Version 2:
//Multiplies matrices assuming matries entries are grouped together
//i.e. [a00, b00, c00, a01, b01, c01 ... ]
//
template<typename myPolicy>
void multiplyVer2(const double * const RAJA_RESTRICT A, const double * const RAJA_RESTRICT B,
                   double * const RAJA_RESTRICT C, const Index_type N);

//
//Version 3
//Layout agnostic function
template<typename myPolicy, typename T>
void multiplyView(const T RAJA_RESTRICT Aview, const T RAJA_RESTRICT Bview,
                  T const RAJA_RESTRICT Cview, const Index_type N);


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA batched matrix multiplication example...\n";

  double myMin;
  srand(time(NULL));

  //Stores data in layout 1
  double * A = memoryManager::allocate<double>(NCOLS*NROWS*NMAT);
  double * B = memoryManager::allocate<double>(NCOLS*NROWS*NMAT);

  //Stores data in layout 2
  double * Al2 = memoryManager::allocate<double>(NCOLS*NROWS*NMAT);
  double * Bl2 = memoryManager::allocate<double>(NCOLS*NROWS*NMAT);

  //Allocates space for output
  double * C1 = memoryManager::allocate<double>(NCOLS*NROWS*NMAT);
  double * C2 = memoryManager::allocate<double>(NCOLS*NROWS*NMAT);

  //
  //Initialize data
  //
  for(Index_type e=0; e<NMAT; ++e){
    for(Index_type row=0; row<NROWS; ++row){
      for(Index_type col=0; col<NCOLS; ++col){
        A(e,row,col) = rand() % 50 + 1; 
        B(e,row,col) = rand() % 50 + 1; 
        Al2(e,row,col) = A(e,row,col);
        Bl2(e,row,col) = B(e,row,col);
      }
    }                 
  }

  //----------------------------------------------------------------------------//                                                                                         
  //Version 1 with macros
  //----------------------------------------------------------------------------//                                                                                          
  myMin = std::numeric_limits<double>::max();
  for(Index_type i=0; i<NITER; ++i){

    auto start = Clock::now();
    multiplyVer1<KernelPol>(A,B,C1,NMAT);
#if defined(RAJA_ENABLE_CUDA)
    cudaDeviceSynchronize();    
#endif
    auto end = Clock::now();

    double tMin = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();   
    if(tMin < myMin) myMin = tMin;
  }
  std::cout<<"Version 1 with macros min run time : "<<myMin<<" miliseconds"<<std::endl;
  //----------------------------------------------------------------------------//                                                                                          

  //----------------------------------------------------------------------------//                                                                                         
  //Version 1 with views
  //----------------------------------------------------------------------------//                                                                                          
  //Equivalent to indexing via B[col + NCOLS*(row + NROWS*elem)] 
  auto layout = RAJA::make_permuted_layout({{NMAT, NROWS, NCOLS}}, RAJA::as_array<RAJA::Perm<0,1,2> >::get() );
  RAJA::View<double, RAJA::Layout<3, Index_type, 2> > Aview(A,layout);
  RAJA::View<double, RAJA::Layout<3, Index_type, 2> > Bview(B,layout);
  RAJA::View<double, RAJA::Layout<3, Index_type, 2> > Cview(C2,layout);

  myMin = std::numeric_limits<double>::max();
  for(Index_type i=0; i<NITER; ++i){
    
    auto start = Clock::now();
    multiplyView<KernelPol>(Aview,Bview,Cview,NMAT);
#if defined(RAJA_ENABLE_CUDA)
    cudaDeviceSynchronize();    
#endif
    auto end = Clock::now();
    
    double tMin = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); 
    if(tMin < myMin) myMin = tMin;
  }
  std::cout<<"Version 1 with RAJA views min run time : "<<myMin<<" milliseconds"<<std::endl;
  
  //
  //Compare output
  //
  compareOutput(C1, C2, NELEM);


  //----------------------------------------------------------------------------//                                                                                         
  //Version 2 with macros
  //----------------------------------------------------------------------------//                                                                                          
  myMin = std::numeric_limits<double>::max();
  for(Index_type i=0; i<NITER; ++i){

    auto start = Clock::now();
    multiplyVer2<KernelPol>(Al2,Bl2,C1,NMAT);
#if defined(RAJA_ENABLE_CUDA)
    cudaDeviceSynchronize();    
#endif
    auto end = Clock::now();

    double tMin = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();   
    if(tMin < myMin) myMin = tMin;
  }
  std::cout<<"Version 2 with macros min run time : "<<myMin<<" seconds"<<std::endl;
  //----------------------------------------------------------------------------//                                                                                          

  //----------------------------------------------------------------------------//                                                                                         
  //Version 2 with views
  //----------------------------------------------------------------------------//
  auto layout2 = RAJA::make_permuted_layout({{NMAT, NROWS, NCOLS}}, RAJA::as_array<RAJA::Perm<1,2,0> >::get() );
  RAJA::View<double, RAJA::Layout<3, Index_type, 0>> Al2view(Al2,layout2);
  RAJA::View<double, RAJA::Layout<3, Index_type, 0>> Bl2view(Bl2,layout2);
  RAJA::View<double, RAJA::Layout<3, Index_type, 0>> Cl2view(C2,layout2);

  myMin = std::numeric_limits<double>::max();
  for(Index_type i=0; i<NITER; ++i){
    
    auto start = Clock::now();
    multiplyView<KernelPol>(Al2view,Bl2view,Cl2view,NMAT);
#if defined(RAJA_ENABLE_CUDA)
    cudaDeviceSynchronize();    
#endif
    auto end = Clock::now();
    
    double tMin = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); 
    if(tMin < myMin) myMin = tMin;
  }
  std::cout<<"Version 2 with RAJA views min run time : "<<myMin<<" milliseconds"<<std::endl;
  
  //
  //Compare output
  //
  compareOutput(C1, C2, NELEM);
  

//
// Clean up.
//
  memoryManager::deallocate(A);
  memoryManager::deallocate(B);
  memoryManager::deallocate(C1);
  memoryManager::deallocate(C2);

  std::cout << "\n DONE!...\n";

  return 0;
}

template<typename myPolicy>
void multiplyVer1(const double * const RAJA_RESTRICT A, const double * const RAJA_RESTRICT B,
                  double * const RAJA_RESTRICT C1, const Index_type N)
{

  RAJA::forall<myPolicy>(RAJA::RangeSegment(0, N), [=] RAJA_DEVICE (Index_type i) {
      
      C1(i,0,0) = A(i,0,0)*B(i,0,0) + A(i,0,1)*B(i,1,0) + A(i,0,2)*B(i,2,0);
      C1(i,0,1) = A(i,0,0)*B(i,0,1) + A(i,0,1)*B(i,1,1) + A(i,0,2)*B(i,2,1);
      C1(i,0,2) = A(i,0,0)*B(i,0,2) + A(i,0,1)*B(i,1,2) + A(i,0,2)*B(i,2,2);

      C1(i,1,0) = A(i,1,0)*B(i,0,0) + A(i,1,1)*B(i,1,0) + A(i,1,2)*B(i,2,0);
      C1(i,1,1) = A(i,1,0)*B(i,0,1) + A(i,1,1)*B(i,1,1) + A(i,1,2)*B(i,2,1);
      C1(i,1,2) = A(i,1,0)*B(i,0,2) + A(i,1,1)*B(i,1,2) + A(i,1,2)*B(i,2,2);

      C1(i,2,0) = A(i,2,0)*B(i,0,0) + A(i,2,1)*B(i,1,0) + A(i,2,2)*B(i,2,0);
      C1(i,2,1) = A(i,2,0)*B(i,0,1) + A(i,2,1)*B(i,1,1) + A(i,2,2)*B(i,2,1);
      C1(i,2,2) = A(i,2,0)*B(i,0,2) + A(i,2,1)*B(i,1,2) + A(i,2,2)*B(i,2,2);
    });
}


template<typename myPolicy>
void multiplyVer2(const double * const RAJA_RESTRICT Al2, const double * const RAJA_RESTRICT Bl2,
                  double * const RAJA_RESTRICT C2, const Index_type N)
{

  RAJA::forall<myPolicy>(RAJA::RangeSegment(0, N), [=] RAJA_DEVICE (Index_type i) {

      C2(i,0,0) = Al2(i,0,0)*Bl2(i,0,0) + Al2(i,0,1)*Bl2(i,1,0) + Al2(i,0,2)*Bl2(i,2,0);
      C2(i,0,1) = Al2(i,0,0)*Bl2(i,0,1) + Al2(i,0,1)*Bl2(i,1,1) + Al2(i,0,2)*Bl2(i,2,1);
      C2(i,0,2) = Al2(i,0,0)*Bl2(i,0,2) + Al2(i,0,1)*Bl2(i,1,2) + Al2(i,0,2)*Bl2(i,2,2);

      C2(i,1,0) = Al2(i,1,0)*Bl2(i,0,0) + Al2(i,1,1)*Bl2(i,1,0) + Al2(i,1,2)*Bl2(i,2,0);
      C2(i,1,1) = Al2(i,1,0)*Bl2(i,0,1) + Al2(i,1,1)*Bl2(i,1,1) + Al2(i,1,2)*Bl2(i,2,1);
      C2(i,1,2) = Al2(i,1,0)*Bl2(i,0,2) + Al2(i,1,1)*Bl2(i,1,2) + Al2(i,1,2)*Bl2(i,2,2);

      C2(i,2,0) = Al2(i,2,0)*Bl2(i,0,0) + Al2(i,2,1)*Bl2(i,1,0) + Al2(i,2,2)*Bl2(i,2,0);
      C2(i,2,1) = Al2(i,2,0)*Bl2(i,0,1) + Al2(i,2,1)*Bl2(i,1,1) + Al2(i,2,2)*Bl2(i,2,1);
      C2(i,2,2) = Al2(i,2,0)*Bl2(i,0,2) + Al2(i,2,1)*Bl2(i,1,2) + Al2(i,2,2)*Bl2(i,2,2);
    });
}

template<typename myPolicy, typename T>
void multiplyView(const T RAJA_RESTRICT Aview, const T RAJA_RESTRICT Bview,
                  T const RAJA_RESTRICT Cview, const Index_type N)
{
  
  RAJA::forall<myPolicy>(RAJA::RangeSegment(0, N), [=] RAJA_DEVICE (Index_type i) {
      
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

//
// Compare output
//
void compareOutput(double *C, double *C2, Index_type Nelem)
{

  bool status = true;
  for(Index_type e = 0; e<NMAT; ++e){
    for(Index_type row=0; row<NROWS; ++row){
      for(Index_type col=0; col<NCOLS; ++col){
        const Index_type id = col + NCOLS*(row + NROWS*e);
        double terr = std::abs(C[id] - C2[id]);
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



