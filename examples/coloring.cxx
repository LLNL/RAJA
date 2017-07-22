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
#include <algorithm>
#include <iostream>
#include <initializer_list>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/index/RangeSegment.hpp"

//Given a checkered board with 4 colors
//This codes picks out the indeces for 
//each color and creates a list of segments
int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  int n         = 4;
  int *A        = new int[n*n];
  
  auto init = {0, 1, 0, 1, 
               2, 3, 2, 3, 
               0, 1, 0, 1,
               2, 3, 2, 3};
  
  std::copy(init.begin(), init.end(), A);
 

  RAJA::IndexSet colorset;

  // populate IndexSet
  {  
    // helps make address calculations easier
    RAJA::View<int,RAJA::Layout<2>> Aview(A, 4, 4);
    
    // buffer used for intermediate indicy storage
    auto idx = RAJA::allocate_aligned_type<RAJA::Index_type>(64, n * n / 4);
  
    // iterate over each dimension (D=2 for this example)
    for (int xdim : {0, 1}) {
      for (int ydim : {0, 1}) {
        
        RAJA::Index_type count = 0;
        
        // iterate over each extent in each dimension, incrementing by two to
        // safely advance over neighbors
        for (int xiter = xdim; xiter < n; xiter += 2) {
          for (int yiter = ydim; yiter < n; yiter += 2) {
            
            // add the computed index to the buffer
            idx[count] = std::distance(A, std::addressof(Aview(xiter, yiter)));
            ++count;
          }
        }

        // insert the indicies added from the buffer as a new ListSegment
        colorset.push_back(RAJA::ListSegment(idx, count));
      }
    }
    
    // clear temporary buffer
    RAJA::free_aligned(idx);
  }

  using ColorPolicy = RAJA::IndexSet::ExecPolicy<RAJA::seq_segit, RAJA::seq_exec>;
  
  RAJA::forall<ColorPolicy>(colorset, [&](int idx) {
      printf("A[%d] = %d\n", idx, A[idx]);
    });
  
  return 0;
}
