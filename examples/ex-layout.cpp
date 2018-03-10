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

#include "RAJA/RAJA.hpp"
#include "memoryManager.hpp"

/*
 *  Layout and View Examples
 *
 *
 *  RAJA features shown:
 *    -  Layout
 *    -  View 
 *    - `forall` loop iteration template method
 *    -  basic usage of the layout object
 */

//
// Function to check result
//

template<typename T>
void printMat(T A, int nRows, int nCols);

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

//
//Define dimension
//
  const int dim = 3;

//
// Define stride length
//  
  const int Istride = 2;
  const int Jstride = 3;
  const int Kstride = 4;

//
//Define range segment
//
  RAJA::RangeSegment Irange(0,Istride);
  RAJA::RangeSegment Jrange(0,Jstride);
  RAJA::RangeSegment Krange(0,Kstride);

//-------------------------------------------//
//
// 1. The following example illustrates how a
//    layout may be used to used to abstract
//    index calculations.
//-------------------------------------------//

//  
// Here we create a 3-d layout object with stride dimensions
// IStride, JStride, Kstride yielding IStride x JStride x Kstride
// unique indices. 
//
  RAJA::Layout<dim> layout(Kstride,Jstride,Istride);

//Layout has a method to map from a multi-index space to
//a linear space.
  int i=1, j=2, k=3;

//Equivalently: int id = i + Istride*(j + Jstride*k)
  int id = layout(k,j,i);

//Print result
  std::cout<<"\nLayout results 1: "<<std::endl;
  std::cout<<"id = "<<id<<" expected: "<< i + Istride*(j + Jstride*k)<<std::endl;
  
//
//We may retrive the inverse of the mapping via the toIndices method
//  
  int ii, jj, kk;
  layout.toIndices(id, kk, jj, ii);
  std::cout<<"("<<ii<<","<<jj<<","<<kk<<")"<<" expected"<<" ("<<i<<","<<j<<","<<k<<")"<<std::endl;

  std::cout<<"---------------------------------------------------------------------------------- \n"<<std::endl;

//We may also permute the layout via the following function call
  RAJA::Layout<dim> KJI_layout = RAJA::make_permuted_layout({Istride,Jstride,Kstride}, RAJA::as_array<RAJA::Perm<2,1,0>>::get());
  
//Print results
  int id2 = KJI_layout(k,j,i);
  std::cout<<"Layout results 2: "<<std::endl;
  std::cout<<"id2 = "<<id2<<" expected: "<< k + Istride*(j + Jstride*i)<<std::endl;

//
//We may retrive the inverse of the mapping via the toIndices method
//    
  layout.toIndices(id, kk, jj, ii);
  std::cout<<"("<<ii<<","<<jj<<","<<kk<<")"<<" expected"<<" ("<<i<<","<<j<<","<<k<<")"<<std::endl;
  std::cout<<"---------------------------------------------------------------------------------- \n"<<std::endl;
  
//-------------------------------------------//
//
// 2. The RAJA::View decouples memmory from layout
//    simplifying multidimensional indexing on a
//    one-dimensional array. 
//-------------------------------------------//
  
//
// Allocate 2D arrays
//
  int array_len = Istride*Jstride;
  int* A = new int[array_len];

//Dimension of the matrix
  const int matDim = 2;

//Here we assume an A{i,j} indexing where i is the fastest
  RAJA::View<int, RAJA::Layout<matDim> > Aview(A, Istride, Jstride);
  
  using NESTED_EXEC_POL = RAJA::nested::Policy<
    RAJA::nested::For<1, RAJA::seq_exec>,
    RAJA::nested::For<0, RAJA::seq_exec>>;

//
//In this example we enumrate the entries of a matrix, in a row major fashion.
//
  RAJA::nested::forall(NESTED_EXEC_POL{},
                       RAJA::make_tuple(Irange, Jrange), [=] (int i, int j){

                         Aview(i,j) = i + Istride*j; 
                         
                       });

  std::cout<<"Matrix should be enumerated in a row major fashion"<<std::endl;
  printMat(Aview, Istride, Jstride);
    
  
//
// Clean up. 
//
  delete[] A;
  std::cout << "\n DONE!...\n";

  return 0;
}

template<typename T>
void printMat(T A, int nRows, int nCols)
{

  for(int r=0; r<nRows; ++r)
    {
      for(int c=0; c<nCols; ++c)
        {
          std::cout<<A(r,c)<<" ";
        }
      std::cout<<""<<std::endl;
    }

  
}
