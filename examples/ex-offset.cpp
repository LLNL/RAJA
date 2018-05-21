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

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "RAJA/RAJA.hpp"
#include "memoryManager.hpp"

/*
 *  Offset example
 *
 *  Example applies a ``five-box stencil" to
 *  interior cells of a lattice. A five-box stencil
 *  accumulates values of a box and its four neighbors.
 *  The resulting values are used to populate the
 *  interior cells of a second lattice.
 *  We assume N x N interior nodes and a padded edge
 *  of zeros for a lattice of size (N + 2) x (N + 2).
 *
 *  In the case of N = 2, the first lattice is generated
 *  to take the form
 *
 *  -----------------
 *  | 0 | 0 | 0 | 0 |
 *  -----------------
 *  | 0 | 1 | 1 | 0 |
 *  -----------------
 *  | 0 | 1 | 1 | 0 |
 *  -----------------
 *  | 0 | 0 | 0 | 0 |
 *  -----------------
 *
 *  post the stencil computation, the second lattice takes
 *  the form of
 *
 *  -----------------
 *  | 0 | 0 | 0 | 0 |
 *  -----------------
 *  | 0 | 3 | 3 | 0 |
 *  -----------------
 *  | 0 | 3 | 3 | 0 |
 *  -----------------
 *  | 0 | 0 | 0 | 0 |
 *  -----------------
 *
 * We simplify computing interior index locations by
 * using RAJA::make_offset_layout and RAJA::Views.
 * RAJA::make_offset_layout enables developers to adjust
 * the enumeration of values in an array. Here we
 * choose to enumerate the lattice in the following manner:
 *
 *  ---------------------------------------
 *  | (-1, 2) | (0, 2)  | (1, 2)  | (2, 2)|
 *  ---------------------------------------
 *  | (-1, 1) | (0, 1)  | (1, 1)  | (2, 1) |
 *  ---------------------------------------
 *  | (-1, 0) | (0, 0)  | (1, 0)  | (2, 0) |
 *  ---------------------------------------
 *  | (-1,-1) | (0, -1) | (1, -1) | (2, -1)|
 *  ---------------------------------------
 *
 *  Notably (0, 0) corresponds to the bottom left
 *  corner of the region we wish to apply our stencil to.
 *
 *  RAJA features shown:
 *    - `forall` loop iteration template method
 *    -  Offset-layouts for RAJA Views
 *    -  Index range segment
 *    -  Execution policies
 */

/*
 * Define number of threads in x and y dimensions of a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
#define CUDA_BLOCK_SIZE 16
#endif

// C-Style macros with offsets
const int offset = -1;
#define lattice0(row, col) lattice0[(col - offset) + (N + 2) * (row - offset)]
#define lattice_ref(row, col) \
  lattice_ref[(col - offset) + (N + 2) * (row - offset)]

//
// Functions for printing and checking results
//
void printBox(int* Box, int boxN);
void checkResult(int* compBox, int* refBox, int boxSize);

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA box-stencil example...\n";

  //
  // Define num of interior cells in rows/cols
  //
  const int N = 4;

  //
  // Define num of cells in rows/cols of the lattice
  //
  const int latticeN = N + 2;

  //
  // Define total num of cells in a lattice
  //
  const int latticeSize = latticeN * latticeN;

  //
  // Allocate and initialize lattice
  //
  int* lattice0 = memoryManager::allocate<int>(latticeSize * sizeof(int));
  int* lattice1 = memoryManager::allocate<int>(latticeSize * sizeof(int));
  int* lattice_ref = memoryManager::allocate<int>(latticeSize * sizeof(int));

  std::memset(lattice0, 0, latticeSize * sizeof(int));
  std::memset(lattice1, 0, latticeSize * sizeof(int));
  std::memset(lattice_ref, 0, latticeSize * sizeof(int));

  //
  // C-Style intialization
  //
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      lattice0(row, col) = 1;
    }
  }

  // prints intial lattice
  // printLattice(lattice0,latticeN);
  //----------------------------------------------------------------------------//

  std::cout << "\n Running C-version of five-lattice-stencil...\n";
  //
  // Perform five-box stencil sum
  //
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {

      lattice_ref(row, col) = lattice0(row, col) + lattice0(row - 1, col)
                              + lattice0(row + 1, col) + lattice0(row, col - 1)
                              + lattice0(row, col + 1);
    }
  }

  // printLattice(lattice_ref,latticeN);
  //----------------------------------------------------------------------------//

  //
  // RAJA versions
  //

  //
  // Create loop bounds
  //
  RAJA::RangeSegment col_range(0, N);
  RAJA::RangeSegment row_range(0, N);

  //
  // Specify the dimension of the lattice
  //
  const int DIM = 2;

  // In the following snippet of code we introduce an offset layout and view
  // to simplify multidimensional indexing.
  // An offset layout is constructed by using the make_offset_layout method.
  // The first argument of the layout is the coordinates of the bottom left
  // corner,
  // and the second array are the cordinates of the top right corner of our
  // shifted
  // lattice.
  //
  RAJA::OffsetLayout<DIM> layout = RAJA::make_offset_layout<DIM>({{-1, -1}}, {{N, N}});    
  RAJA::View<int, RAJA::OffsetLayout<DIM>> lattice0view(lattice0, layout);
  RAJA::View<int, RAJA::OffsetLayout<DIM>> lattice1view(lattice1, layout);

  //----------------------------------------------------------------------------//
  std::cout << "\n Running sequential five-box-stencil (RAJA-Kernel - "
               "sequential)...\n";
  using NESTED_EXEC_POL = RAJA::
      KernelPolicy<RAJA::statement::
        For<1, RAJA::seq_exec,  // row          
          RAJA::statement::For<0,RAJA::seq_exec,  // col                           
                           RAJA::statement::Lambda<0>>>>;

  RAJA::kernel<NESTED_EXEC_POL>(RAJA::make_tuple(col_range, row_range),
                                [=](int col, int row) {

                                  lattice1view(row, col) =
                                      lattice0view(row, col)
                                      + lattice0view(row - 1, col)
                                      + lattice0view(row + 1, col)
                                      + lattice0view(row, col - 1)
                                      + lattice0view(row, col + 1);
                                });

  // printLattice(lattice1,latticeN);
  checkResult(lattice1, lattice_ref, latticeSize);

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::cout << "\n Running sequential five-box-stencil (RAJA-Kernel - omp "
               "parallel for)...\n";
  using NESTED_EXEC_POL2 = RAJA::
    KernelPolicy<RAJA::statement::
      For<1, RAJA::omp_parallel_for_exec,  // row          
          RAJA::statement::For<0, RAJA::seq_exec,  // col             
                               RAJA::statement::Lambda<0>>>>;

  RAJA::kernel<NESTED_EXEC_POL2>(RAJA::make_tuple(col_range, row_range),
                                 [=](int col, int row) {

                                   lattice1view(row, col) =
                                       lattice0view(row, col)
                                       + lattice0view(row - 1, col)
                                       + lattice0view(row + 1, col)
                                       + lattice0view(row, col - 1)
                                       + lattice0view(row, col + 1);
                                 });

  // printLattice(lattice1,N);
  checkResult(lattice1, lattice_ref, latticeSize);
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

  std::cout << "\n Running sequential five-box-stencil (RAJA-Kernel - "
               "cuda)...\n";

  using NESTED_EXEC_POL3 = RAJA::
    KernelPolicy<RAJA::statement::
      CudaKernel<RAJA::statement::
                 For<1, RAJA::cuda_threadblock_exec<CUDA_BLOCK_SIZE>,  // row
                     RAJA::statement::For<0, RAJA::cuda_threadblock_exec<CUDA_BLOCK_SIZE>,  // col
                                          RAJA::statement::Lambda<0>>>>>;
                                                     

  RAJA::kernel<NESTED_EXEC_POL3>(RAJA::make_tuple(col_range, row_range),
                                 [=] RAJA_DEVICE(int col, int row) {

                                   lattice1view(row, col) =
                                       lattice0view(row, col)
                                       + lattice0view(row - 1, col)
                                       + lattice0view(row + 1, col)
                                       + lattice0view(row, col - 1)
                                       + lattice0view(row, col + 1);
                                 });

  // printLattice(lattice1,latticeN);
  checkResult(lattice1, lattice_ref, latticeSize);
//----------------------------------------------------------------------------//
#endif

  //
  // Clean up.
  //
  memoryManager::deallocate(lattice0);
  memoryManager::deallocate(lattice1);
  memoryManager::deallocate(lattice_ref);

  std::cout << "\n DONE!...\n";
  return 0;
}

//
// Print Lattice
//
void printLattice(int* lattice, int latticeN)
{
  std::cout << std::endl;
  for (int row = 0; row < latticeN; ++row) {
    for (int col = 0; col < latticeN; ++col) {

      const int id = col + latticeN * row;
      std::cout << lattice[id] << " ";
    }
    std::cout << " " << std::endl;
  }
  std::cout << std::endl;
}

//
// Check Result
//
void checkResult(int* compLattice, int* refLattice, int len)
{

  bool pass = true;

  for (int i = 0; i < len; ++i) {
    if (compLattice[i] != refLattice[i]) pass = false;
  }

  if (pass) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
}
