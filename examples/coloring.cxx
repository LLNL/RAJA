//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/index/RangeSegment.hpp"


//Colring Example
int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  int n         = 4;
  int entries   = n*n;
  int numCells  = 4;
  int *A        = new int[entries];
  int *colorSet = new int[entries];

  A[0]  = 0; A[1]  = 0; A[2]   = 1; A[3]  = 1;
  A[4]  = 0; A[5]  = 0; A[6]   = 1; A[7]  = 1; 
  A[8]  = 2; A[9]  = 2; A[10]  = 3; A[11] = 3; 
  A[12] = 2; A[13] = 2; A[14]  = 3; A[15] = 3;

  colorSet[0]  = 0; colorSet[1]   = 1; colorSet[2]   = 4;  colorSet[3]  = 5;
  colorSet[4]  = 2; colorSet[5]   = 3; colorSet[6]   = 3;  colorSet[7]  = 7;
  colorSet[8]  = 8; colorSet[9]   = 9; colorSet[10]  = 12; colorSet[11] = 13;
  colorSet[12] = 10; colorSet[13] = 11; colorSet[14] = 14; colorSet[15] = 15;

  
  RAJA::RangeStrideSegment myRange(0,numCells,4);


  //How do I use RangeStrideSegment?

  //RAJA::forall<RAJA::seq_exec>(myRange,[=](int i){
  //std::cout<<"i = "<<i<<std::endl;
  //});


  



 
  //RAJA::IndexSet segments;
  //int elems [] = {0,1,2,3,4,5,6,7};                  
  //RAJA::buildIndexSetAligned(segments,col_index,cInd_ct);    

  delete[] A, colorSet;

  return 0;
}
