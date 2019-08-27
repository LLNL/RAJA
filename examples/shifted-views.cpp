//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{


  int N = 10;
  int *a = memoryManager::allocate<int>(N);

  const int DIM = 1;
  RAJA::OffsetLayout<DIM> layout = RAJA::make_offset_layout<DIM>({{0}},{{9}});

  RAJA::View<int, RAJA::OffsetLayout<DIM>> A(a, layout);


  for(int i=0; i<N; ++i) {
    A(i) = i + 1;
  }


  printf("Original view \n");
  for(int i=0; i<N; ++i) {
    printf("%d ",A(i));
  }printf("\n");


  printf("Shifted view \n");
  RAJA::View<int, RAJA::OffsetLayout<DIM>> B =  A.shift<DIM>({{10}},{{19}});

  for(int i=10; i<20; ++i) {
    printf("%d ",B(i));
  }printf("\n");

  memoryManager::deallocate(a);

  return 0;
}
