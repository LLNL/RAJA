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

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"
#include "RAJA/util/Timer.hpp"

#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 256;
#endif

//
// Define dimensionality of matrices
//
const int DIM = 2;

//
// Define num rows/cols in matrix
//

//Matrix A size : N x M
//Matrix B size : M x P
//Matrix C size : N x P

const int M = 1536;
const int N = 1536;
const int P = 1536;

// Define TILE dimensions
//
const int TILE_DIM = 32;

//
// Define bounds for inner and outer loops
//
const int inner_Dim0 = TILE_DIM;
const int inner_Dim1 = TILE_DIM;

const int windowIter = (M-1)/TILE_DIM+1;

const int outer_Dim0 = (P-1)/TILE_DIM+1;
const int outer_Dim1 = (N-1)/TILE_DIM+1;

template<typename T, typename U>
void checkResult(T Cview, U C_solview)
{
  //Check result
  bool pass = true;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < P; ++col) {
      
      if(Cview(row, col) != C_solview(row, col)) pass = false;
    }
  }
  
  if(pass){
    printf("Pass! \n");
  }else{
    printf("Fail \n");
  }

}

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA shared memory matrix multiplication example...\n";
  
  int NITER = 5;
  auto timer = RAJA::Timer();
  double minRun;
  
  //
  // Allocate matrix data
  //
  double *A      = memoryManager::allocate<double>(N * M);
  double *B      = memoryManager::allocate<double>(M * P);
  double *C      = memoryManager::allocate<double>(N * P);
  double *C_sol  = memoryManager::allocate<double>(N * P);
  
  //
  // In the following implementations of shared matrix transpose, we
  // use RAJA 'View' objects to access the matrix data. A RAJA view
  // holds a pointer to a data array and enables multi-dimensional indexing
  // into the data.
  //
  RAJA::View<double, RAJA::Layout<DIM>> Aview(A, N, M);
  RAJA::View<double, RAJA::Layout<DIM>> Bview(B, M, P);
  RAJA::View<double, RAJA::Layout<DIM>> Cview(C, N, P);
  RAJA::View<double, RAJA::Layout<DIM>> C_solview(C, N, P);
  
  //
  // Initialize matrix data
  //
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < M; ++col) {
      Aview(row, col) = col;
    }
  }
  
  for(int row = 0; row < M; ++row) {
    for(int col = 0; col < P; ++col) {
      Bview(row, col) = col;
    }
  }
  
  for(int r=0; r<N; ++r){
    for(int c=0; c<P; ++c){
      double dot = 0.0;
      for(int k=0; k<M; ++k){
        dot += Aview(r,k)*Bview(k,c);
      }
      C_solview(r,c) = dot;
    }
  }


  //----------------------------------------------------------------------------//
  std::cout << "\n Running C-version of naive matrix multiplication algorithm...\n";
  std::memset(C, 0, sizeof(double)*N*P);
  
  minRun = std::numeric_limits<double>::max();
  for (int i = 0; i < NITER; ++i) {

    timer.start();
    
    for(int r=0; r<N; ++r){
      for(int c=0; c<P; ++c){
        double dot = 0.0;
        for(int k=0; k<M; ++k){
          dot += Aview(r,k)*Bview(k,c);
        }
        C_solview(r,c) = dot;
      }
    }    
    timer.stop();
    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }
  std::cout<< "\trun time : " << minRun << " seconds" << std::endl;
  checkResult(Cview, C_solview);  

  //----------------------------------------------------------------------------//
  std::cout << "\n Running C-version of shared matrix multiplication with window algorithm...\n";
  std::memset(C, 0, sizeof(double)*N*P);
  
  timer.reset();
  minRun = std::numeric_limits<double>::max();
  for (int i = 0; i < NITER; ++i) {
    
    timer.start();
    //
    // (0) Outer loops to iterate over tiles
    //
    for (int by = 0; by < outer_Dim1; ++by) {
      for (int bx = 0; bx < outer_Dim0; ++bx) {
        
        double aShared[TILE_DIM][TILE_DIM]; // shared memory
        double bShared[TILE_DIM][TILE_DIM]; // shared memory
        double pValue[TILE_DIM][TILE_DIM]; //thread private value
        
        //
        // (1) Initialize thread private value
        //
        for(int ty = 0; ty < inner_Dim1; ++ty){
          for(int tx = 0; tx < inner_Dim0; ++tx){
            pValue[ty][tx] = 0.0;
          }
        }
        
        //Loop to slide window across the matrix
        for(int i = 0; i < windowIter; ++i) {
          
          //
          // (2) Inner loops to load data into the tile
          //
          for (int ty = 0; ty < inner_Dim1; ++ty) {
            for (int tx = 0; tx < inner_Dim0; ++tx) {
              
              int col = bx * TILE_DIM + tx;  // Matrix column index
              int row = by * TILE_DIM + ty;  // Matrix row index

              aShared[ty][tx] = Aview(row, ((i*TILE_DIM+tx) ));
              bShared[ty][tx] = Bview((i*TILE_DIM + ty), col);
              /*              
              if(row < N && ((i*TILE_DIM + tx) < M)){
                aShared[ty][tx] = Aview(row, ((i*TILE_DIM+tx) ));
              }else{
                aShared[ty][tx] = 0.0;
              }
              
              if( col < P && ((i*TILE_DIM + ty) < M) ){
              bShared[ty][tx] = Bview((i*TILE_DIM + ty), col);
              }else{
                bShared[ty][tx] = 0.0;
              }
              */
              
            }
          }
          //Syncthreads
          
          //
          // (3) Matrix mutiply
          //
          for (int ty = 0; ty < inner_Dim1; ++ty) {
            for (int tx = 0; tx < inner_Dim0; ++tx) {
              
              for(int j=0; j<TILE_DIM; ++j){
                pValue[ty][tx] += aShared[ty][j]*bShared[j][tx];
              }
              
            }
          }
          
        }//loop to slide window across matrix
        
        //
        // (4) Write out to global matrix
        //
        for (int ty = 0; ty < inner_Dim1; ++ty) {
          for (int tx = 0; tx < inner_Dim0; ++tx) {
            
            int row = by * TILE_DIM + ty;  // Matrix row index
            int col = bx * TILE_DIM + tx;  // Matrix column index
            
            //if(row < N && col < P)
              Cview(row, col) = pValue[ty][tx];
          }
        }
        
      }
    }
    timer.stop();
    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }
  std::cout<< "\trun time : " << minRun << " seconds" << std::endl;
  checkResult(Cview, C_solview);  
  //----------------------------------------------------------------------------//

  //With scalar values
  std::cout << "\n Running C-version of shared matrix multiplication with window algorithm and pValue as a scalar...\n";
  std::memset(C, 0, sizeof(double)*N*P);

  timer.reset();
  minRun = std::numeric_limits<double>::max();
  for (int i = 0; i < NITER; ++i) {
    
    timer.start();
    //
    // (0) Outer loops to iterate over tiles
    //
    for (int by = 0; by < outer_Dim1; ++by) {
      for (int bx = 0; bx < outer_Dim0; ++bx) {
        
        double aShared[TILE_DIM][TILE_DIM]; // shared memory
        double bShared[TILE_DIM][TILE_DIM]; // shared memory
        //double pValue[TILE_DIM][TILE_DIM]; //thread private value
                
        //
        // (1) Initialize thread private value
        //
        //for(int ty = 0; ty < inner_Dim1; ++ty){
        //for(int tx = 0; tx < inner_Dim0; ++tx){
        //pValue[ty][tx] = 0.0;
        //}
        //}

        //Loop to slide window across the matrix
        for(int i = 0; i < windowIter; ++i) {
          
          //
          // (2) Inner loops to load data into the tile
          //
          for (int ty = 0; ty < inner_Dim1; ++ty) {
            for (int tx = 0; tx < inner_Dim0; ++tx) {
              
              int col = bx * TILE_DIM + tx;  // Matrix column index
              int row = by * TILE_DIM + ty;  // Matrix row index

              aShared[ty][tx] = Aview(row, ((i*TILE_DIM+tx) ));
              bShared[ty][tx] = Bview((i*TILE_DIM + ty), col);
              /*              
              if(row < N && ((i*TILE_DIM + tx) < M)){
                aShared[ty][tx] = Aview(row, ((i*TILE_DIM+tx) ));
              }else{
                aShared[ty][tx] = 0.0;
              }
              
              if( col < P && ((i*TILE_DIM + ty) < M) ){
                bShared[ty][tx] = Bview((i*TILE_DIM + ty), col);
              }else{
                bShared[ty][tx] = 0.0;
              }
              */
              
            }
          }
          //Syncthreads
          
          //
          // (3) Matrix mutiply
          //
          for (int ty = 0; ty < inner_Dim1; ++ty) {
            for (int tx = 0; tx < inner_Dim0; ++tx) {

              int row = by * TILE_DIM + ty;  // Matrix row index
              int col = bx * TILE_DIM + tx;  // Matrix column index
              
              double pValue = 0.0;
              for(int j=0; j<TILE_DIM; ++j){
                pValue += aShared[ty][j]*bShared[j][tx];
              }
              
              ///if(row < N && col < P)
              Cview(row,col) += pValue;
              
            }
          }
          
        }//loop to slide window across matrix
                
      }
    }
    timer.stop();
    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }
  checkResult(Cview, C_solview);
  std::cout<< "\trun time : " << minRun << " seconds" << std::endl;


  //
  // Clean up.
  //
  memoryManager::deallocate(A);
  memoryManager::deallocate(B);
  memoryManager::deallocate(C);
  memoryManager::deallocate(C_sol);

  std::cout << "\n DONE!...\n";

  return 0;
}
