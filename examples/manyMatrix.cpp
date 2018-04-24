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
#if defined (RAJA_ENABLE_CUDA)
#include "cuda.h"
#endif

#include "RAJA/RAJA.hpp"
#include "memoryManager.hpp"

using RAJA::Index_type;
typedef std::chrono::high_resolution_clock Clock;
const int MAT_ENTRIES = 9;
const int NCOLS = 3;
const int NROWS = 3;

#define A(elem, row, col) A[col + row*NCOLS + elem*MAT_ENTRIES]
#define B(elem, row, col) B[col + row*NCOLS + elem*MAT_ENTRIES]
#define C(elem, row, col) C[col + row*NCOLS + elem*MAT_ENTRIES]

#define AComp(elem, row, col) AComp[elem + Nelem*(col + row*NCOLS)]
#define BComp(elem, row, col) BComp[elem + Nelem*(col + row*NCOLS)]
#define CComp(elem, row, col) CComp[elem + Nelem*(col + row*NCOLS)]

#if defined (RAJA_ENABLE_CUDA)
template<typename myPolicy>
void multiplyVer1(const double * const RAJA_RESTRICT A, const double * const RAJA_RESTRICT B,
                 double * const RAJA_RESTRICT C, const size_t Nelem)
{

  RAJA::forall<myPolicy>(RAJA::RangeSegment(0, Nelem), [=] RAJA_DEVICE (Index_type i) {
      
      C(i,0,0) = A(i,0,0)*B(i,0,0) + A(i,0,1)*B(i,1,0) + A(i,0,2)*B(i,2,0);
      C(i,0,1) = A(i,0,0)*B(i,0,1) + A(i,0,1)*B(i,1,1) + A(i,0,2)*B(i,2,1);
      C(i,0,2) = A(i,0,0)*B(i,0,2) + A(i,0,1)*B(i,1,2) + A(i,0,2)*B(i,2,2);

      C(i,1,0) = A(i,1,0)*B(i,0,0) + A(i,1,1)*B(i,1,0) + A(i,1,2)*B(i,2,0);
      C(i,1,1) = A(i,1,0)*B(i,0,1) + A(i,1,1)*B(i,1,1) + A(i,1,2)*B(i,2,1);
      C(i,1,2) = A(i,1,0)*B(i,0,2) + A(i,1,1)*B(i,1,2) + A(i,1,2)*B(i,2,2);

      C(i,2,0) = A(i,2,0)*B(i,0,0) + A(i,2,1)*B(i,1,0) + A(i,2,2)*B(i,2,0);
      C(i,2,1) = A(i,2,0)*B(i,0,1) + A(i,2,1)*B(i,1,1) + A(i,2,2)*B(i,2,1);
      C(i,2,2) = A(i,2,0)*B(i,0,2) + A(i,2,1)*B(i,1,2) + A(i,2,2)*B(i,2,2);
    });
}


//Check for correctness
void compareOutput(double *C, double *CCComp, size_t N);

 
template<typename myPolicy>
void multiplyVer2(const double * const RAJA_RESTRICT AComp, const double * const RAJA_RESTRICT BComp,
                     double * const RAJA_RESTRICT CComp, const size_t Nelem)
{

  RAJA::forall<myPolicy>(RAJA::RangeSegment(0, Nelem), [=] RAJA_DEVICE (Index_type i) {
      
      CComp(i,0,0) = AComp(i,0,0)*BComp(i,0,0) + AComp(i,0,1)*BComp(i,1,0) + AComp(i,0,2)*BComp(i,2,0);
      CComp(i,0,1) = AComp(i,0,0)*BComp(i,0,1) + AComp(i,0,1)*BComp(i,1,1) + AComp(i,0,2)*BComp(i,2,1);
      CComp(i,0,2) = AComp(i,0,0)*BComp(i,0,2) + AComp(i,0,1)*BComp(i,1,2) + AComp(i,0,2)*BComp(i,2,2);

      CComp(i,1,0) = AComp(i,1,0)*BComp(i,0,0) + AComp(i,1,1)*BComp(i,1,0) + AComp(i,1,2)*BComp(i,2,0);
      CComp(i,1,1) = AComp(i,1,0)*BComp(i,0,1) + AComp(i,1,1)*BComp(i,1,1) + AComp(i,1,2)*BComp(i,2,1);
      CComp(i,1,2) = AComp(i,1,0)*BComp(i,0,2) + AComp(i,1,1)*BComp(i,1,2) + AComp(i,1,2)*BComp(i,2,2);

      CComp(i,2,0) = AComp(i,2,0)*BComp(i,0,0) + AComp(i,2,1)*BComp(i,1,0) + AComp(i,2,2)*BComp(i,2,0);
      CComp(i,2,1) = AComp(i,2,0)*BComp(i,0,1) + AComp(i,2,1)*BComp(i,1,1) + AComp(i,2,2)*BComp(i,2,1);
      CComp(i,2,2) = AComp(i,2,0)*BComp(i,0,2) + AComp(i,2,1)*BComp(i,1,2) + AComp(i,2,2)*BComp(i,2,2);
    });
}


template<typename myPolicy, typename T>
void multiplyView(const T RAJA_RESTRICT Aview, const T RAJA_RESTRICT Bview,
                  T const RAJA_RESTRICT Cview, const size_t Nelem)
{


  RAJA::forall<myPolicy>(RAJA::RangeSegment(0, Nelem), [=] RAJA_DEVICE (Index_type i) {

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

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  srand(time(NULL));
  const int Nelem = 25600;
  const int NITER = 10000;
  double myMin, totalTime;

#if defined (RAJA_ENABLE_CUDA)

  double * const RAJA_RESTRICT A = memoryManager::allocate<double>(Nelem*MAT_ENTRIES);
  double * const RAJA_RESTRICT B = memoryManager::allocate<double>(Nelem*MAT_ENTRIES);
  double * const RAJA_RESTRICT C = memoryManager::allocate<double>(Nelem*MAT_ENTRIES);

  double * const RAJA_RESTRICT AComp = memoryManager::allocate<double>(Nelem*MAT_ENTRIES);
  double * const RAJA_RESTRICT BComp = memoryManager::allocate<double>(Nelem*MAT_ENTRIES);
  double * const RAJA_RESTRICT CComp = memoryManager::allocate<double>(Nelem*MAT_ENTRIES);

  //Loop over elements and initialize
  //--------------------------------------------
  for(Index_type e=0; e<Nelem; ++e)
    {
      for(Index_type row=0; row<NROWS; ++row)
        {
          for(Index_type col=0; col<NROWS; ++col)
            {
              A(e,row,col) = rand() % 50 + 1;
              B(e,row,col) = rand() % 50 + 1;
              AComp(e,row,col) = A(e,row,col);
              BComp(e,row,col) = B(e,row,col);
            }
        }
    }
  //--------------------------------------------

  //=================================
  //Matrix Multiplication with Macros
  //==================================

  //-----------------
  //Version 1: 
  //Matrices are stored in a contigous manner [a00,a01,a02,a10,a11,a12, ... 
  //-----------------
  myMin = std::numeric_limits<double>::max();
  totalTime = 0;
  for(int i=0; i<NITER; ++i){

    auto start = Clock::now();
    multiplyVer1<RAJA::cuda_exec<256>>(A,B,C,Nelem);
    cudaDeviceSynchronize();
    auto end = Clock::now();
    
    double tMin = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    totalTime += tMin;
    if(tMin < myMin) myMin = tMin;
  }
  std::cout<<"Contiguous matrices AVG Run Time "<<totalTime/NITER<< " nano-seconds"<<std::endl;
  std::cout<<"Contiguous matrices Min Run Time "<<myMin<<" nano-seconds"<<std::endl;


  //----------------------------------------------------------

  //-----------------
  //Version 2: 
  //Matrices entries are stored contigous [a00, b00, c00, d00,... 
  //-----------------
  myMin = std::numeric_limits<double>::max();
  totalTime = 0;
  for(int i=0; i<NITER; ++i){

    auto start = Clock::now();
    multiplyVer2<RAJA::cuda_exec<256>>(AComp,BComp,CComp,Nelem);
    cudaDeviceSynchronize();
    auto end = Clock::now();

    double tMin = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    totalTime += tMin;
    if(tMin < myMin) myMin = tMin;
  }
  std::cout<<"Contigous Matrix Entries AVG Run Time "<<totalTime/NITER<<" nano-seconds"<<std::endl;
  std::cout<<"Contiguous Matrix Entries Min Run Time "<<myMin<<" nano-seconds"<<std::endl;
  //----------------------------------------------------------

  //
  compareOutput(C,CComp, Nelem);
  std::cout<<"\n \n \n"<<std::endl;


  //=================================
  //Repeat experiment with RAJA views
  //==================================

  // Version 1 with Views
  myMin = std::numeric_limits<double>::max();
  totalTime = 0;
  RAJA::View<double, RAJA::Layout<3,Index_type, 0> > Aview(A,Nelem, NROWS, NCOLS);
  RAJA::View<double, RAJA::Layout<3,Index_type, 0> > Bview(B,Nelem, NROWS, NCOLS);
  RAJA::View<double, RAJA::Layout<3,Index_type, 0> > Cview(C,Nelem, NROWS, NCOLS);

  for(int i=0; i<NITER; ++i){

    auto start = Clock::now();
    multiplyView<RAJA::cuda_exec<256>>(Aview,Bview,Cview,Nelem);
    cudaDeviceSynchronize();
    auto end = Clock::now();

    double tMin = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    totalTime += tMin;
    if(tMin < myMin) myMin = tMin;
  }
  std::cout<<"Contigous Matrices with Views AVG Run Time "<<totalTime/NITER<<" nano-seconds"<<std::endl;
  std::cout<<"Contiguous Matrices with Views Min Run Time "<<myMin<<" nano-seconds"<<std::endl;


  //---------------------------------------------------------- 
  //Version 2 with views ? 
  myMin = std::numeric_limits<double>::max();
  totalTime = 0;

  //Question: How do I permute this in order to get the same striding as : 
  //#define AComp(elem, row, col) AComp[elem + Nelem*(col + row*NCOLS)]  ? 
  
  RAJA::View<double, RAJA::Layout<3> > AviewP(A, Nelem, NROWS, NCOLS);
  RAJA::View<double, RAJA::Layout<3> > BviewP(B, Nelem, NROWS, NCOLS);
  RAJA::View<double, RAJA::Layout<3> > CviewP(C, Nelem, NROWS, NCOLS);

  for(int i=0; i<NITER; ++i){

    auto start = Clock::now();
    multiplyView<RAJA::cuda_exec<256>>(AviewP,BviewP,CviewP,Nelem);
    cudaDeviceSynchronize();
    auto end = Clock::now();

    double tMin = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    totalTime += tMin;
    if(tMin < myMin) myMin = tMin;
  }
  std::cout<<"Contigous Matrix Entries AVG Run Time "<<totalTime/NITER<<" nano-seconds"<<std::endl;
  std::cout<<"Contiguous Matrix Entries Min Run Time "<<myMin<<" nano-seconds"<<std::endl;
  //----------------------------------------------------------


  compareOutput(C,CComp, Nelem);
  std::cout<<"\n \n \n"<<std::endl;

#endif    
  
//
// Clean up. 
//
  std::cout << "\n DONE!...\n";

  return 0;
}



void compareOutput(double *C, double *CComp, size_t Nelem)
{

  bool status = true;
  for(Index_type e = 0; e<Nelem; ++e)
    {      
      for(Index_type r=0; r<NROWS; ++r)
        {
          for(Index_type c=0; c<NROWS; ++c)
            {
              double terr = std::abs(C(e,r,c) - CComp(e,r,c));
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

