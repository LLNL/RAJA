//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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
 *  Offset Layout example
 *
 *  This example applies a five-cell stencil to the
 *  interior cells of a lattice and stores the 
 *  resulting sums in a second lattice of equal size.
 *
 *  The five-cell stencil accumulates values of a cell 
 *  and its four neighbors. Assuming the cells of a 
 *  lattice may be accessed through a row/col fashion, 
 *  the stencil may be expressed as the following sum
 * 
 *  output_lattice(row, col)
 *         = input_lattice(row, col)
 *         + input_lattice(row - 1, col) + input_lattice(row + 1, col)
 *         + input_lattice(row, col - 1) + input_lattice(row, col + 1)
 *
 *  We assume a lattice has N x N interior nodes 
 *  and a padded edge of zeros for a lattice
 *  of size (N_r + 2) x (N_c + 2).  
 *
 *  In the case of N = 3, the input lattice generated
 *  takes the form
 *
 *  ---------------------
 *  | 0 | 0 | 0 | 0 | 0 |
 *  ---------------------
 *  | 0 | 1 | 1 | 1 | 0 |
 *  ---------------------
 *  | 0 | 1 | 1 | 1 | 0 |
 *  ---------------------
 *  | 0 | 1 | 1 | 1 | 0 |
 *  ---------------------
 *  | 0 | 0 | 0 | 0 | 0 |
 *  ---------------------
 *
 *  after the computation, we expect the output
 *  lattice to take the form
 *
 *  ---------------------
 *  | 0 | 0 | 0 | 0 | 0 |
 *  ---------------------
 *  | 0 | 3 | 4 | 3 | 0 |
 *  ---------------------
 *  | 0 | 4 | 5 | 4 | 0 |
 *  ---------------------
 *  | 0 | 3 | 4 | 3 | 0 |
 *  ---------------------
 *  | 0 | 0 | 0 | 0 | 0 |
 *  ---------------------
 *
 * In this example, we use RAJA's make_offset_layout
 * method and view object to simplify applying
 * the stencil to interior cells.
 * The make_offset_layout method enables developers
 * to create layouts which offset
 * the enumeration of values in an array. Here we
 * choose to enumerate the lattice in the following manner:
 *
 *  --------------------------------------------------
 *  | (-1, 3) | (0, 3)  | (1, 3)  | (2, 3)  | (3, 3)  |
 *  --------------------------------------------------
 *  | (-1, 2) | (0, 2)  | (1, 2)  | (2, 2)  | (3, 2)  |
 *  --------------------------------------------------
 *  | (-1, 1) | (0, 1)  | (1, 1)  | (2, 1)  | (3, 1)  |
 *  --------------------------------------------------
 *  | (-1, 0) | (0, 0)  | (1, 0)  | (2, 0)  | (3, 0)  |
 *  ---------------------------------------------------
 *  | (-1,-1) | (0, -1) | (1, -1) | (2, -1) | (3, -1) |
 *  ---------------------------------------------------
 *
 *  Notably (0, 0) corresponds to the bottom left
 *  corner of the region to which we wish to apply stencil.
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

//
// Functions for printing and checking results
//
void printLattice(int* lattice, int N_r, int N_c);
void checkResult(int* compLattice, int* refLattice, int totCells);

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA five-cell stencil example...\n";

//
// Define num of interior cells in row/cols in a lattice
//
  const int N_r = 3;
  const int N_c = 3;

//
// Define total num of cells in rows/cols in a lattice
//
  const int totCellsInRow = N_r + 2;
  const int totCellsInCol = N_c + 2;

//
// Define total num of cells in a lattice
//
  const int totCells = totCellsInRow * totCellsInCol;

//
// Allocate and initialize lattice
//
  int* input_lattice = memoryManager::allocate<int>(totCells * sizeof(int));
  int* output_lattice = memoryManager::allocate<int>(totCells * sizeof(int));
  int* lattice_ref = memoryManager::allocate<int>(totCells * sizeof(int));

  std::memset(input_lattice, 0, totCells * sizeof(int));
  std::memset(output_lattice, 0, totCells * sizeof(int));
  std::memset(lattice_ref, 0, totCells * sizeof(int));

//
// C-Style intialization
//
  for (int row = 1; row <= N_r; ++row) {
    for (int col = 1; col <= N_c; ++col) {
      int id = col + totCellsInCol * row;
      input_lattice[id] = 1;
    }
  }
// printLattice(input_lattice, totCellsInRow, totCellsInCol);

//
// Generate reference solution
//
  for (int row = 1; row <= N_r; ++row) {
    for (int col = 1; col <= N_c; ++col) {

      int id = col + totCellsInCol * row;
      lattice_ref[id] = input_lattice[id] + input_lattice[id + 1]
                        + input_lattice[id - 1]
                        + input_lattice[id + totCellsInCol]
                        + input_lattice[id - totCellsInCol];
    }
  }
// printLattice(lattice_ref, totCellsInRow, totCellsInCol);

//----------------------------------------------------------------------------//

//
// Create loop bounds
//
  RAJA::RangeSegment col_range(0, N_r);
  RAJA::RangeSegment row_range(0, N_c);

// The following code illustrates pairing an offset layout and a RAJA view
// object to simplify multidimensional indexing.
// An offset layout is constructed by using the make_offset_layout method.
// The first argument of the layout is an array object with the coordinates of
// the bottom left corner of the lattice, and the second argument is an array 
// object of the coordinates of the top right corner.
// The example uses double braces to initiate the array object and its
// subobjects.
//
  const int DIM = 2;

  RAJA::OffsetLayout<DIM> layout =
      RAJA::make_offset_layout<DIM>({{-1, -1}}, {{N_r, N_c}});

  RAJA::View<int, RAJA::OffsetLayout<DIM>> input_latticeView(input_lattice, layout);
  RAJA::View<int, RAJA::OffsetLayout<DIM>> output_latticeView(output_lattice, layout);

//----------------------------------------------------------------------------//

  std::cout << "\n Running five-cell stencil (RAJA-Kernel - "
               "sequential)...\n";

  using NESTED_EXEC_POL1 =
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::seq_exec,    // row
        RAJA::statement::For<0, RAJA::seq_exec,  // col
          RAJA::statement::Lambda<0>
        >
      >  
    >;  

  RAJA::kernel<NESTED_EXEC_POL1>(RAJA::make_tuple(col_range, row_range),
                                 [=](int col, int row) {

                                   output_latticeView(row, col) =
                                       input_latticeView(row, col)
                                       + input_latticeView(row - 1, col)
                                       + input_latticeView(row + 1, col)
                                       + input_latticeView(row, col - 1)
                                       + input_latticeView(row, col + 1);
                                 });

  //printLattice(lattice_ref, totCellsInRow, totCellsInCol);
  checkResult(output_lattice, lattice_ref, totCells);

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::cout << "\n Running five-cell stencil (RAJA-Kernel - omp "
               "parallel for)...\n";

  using NESTED_EXEC_POL2 = 
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::omp_parallel_for_exec, // row
        RAJA::statement::For<0, RAJA::seq_exec,            // col
          RAJA::statement::Lambda<0>
        > 
      > 
    >;

  RAJA::kernel<NESTED_EXEC_POL2>(RAJA::make_tuple(col_range, row_range),
                                 [=](int col, int row) {

                                   output_latticeView(row, col) =
                                       input_latticeView(row, col)
                                       + input_latticeView(row - 1, col)
                                       + input_latticeView(row + 1, col)
                                       + input_latticeView(row, col - 1)
                                       + input_latticeView(row, col + 1);
                                 });

  //printLattice(lattice_ref, totCellsInRow, totCellsInCol);
  checkResult(output_lattice, lattice_ref, totCells);
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

  std::cout << "\n Running five-cell stencil (RAJA-Kernel - "
               "cuda)...\n";

  using NESTED_EXEC_POL3 =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
        RAJA::statement::For<1, RAJA::cuda_block_x_loop, //row
          RAJA::statement::For<0, RAJA::cuda_thread_x_loop, //col
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;                                                     

  RAJA::kernel<NESTED_EXEC_POL3>(RAJA::make_tuple(col_range, row_range),
                                 [=] RAJA_DEVICE(int col, int row) {

                                   output_latticeView(row, col) =
                                       input_latticeView(row, col)
                                       + input_latticeView(row - 1, col)
                                       + input_latticeView(row + 1, col)
                                       + input_latticeView(row, col - 1)
                                       + input_latticeView(row, col + 1);
                                 });

  //printLattice(output_lattice, totCellsInRow, totCellsInCol);
  checkResult(output_lattice, lattice_ref, totCells);
#endif

//----------------------------------------------------------------------------//

//
// Clean up.
//
  memoryManager::deallocate(input_lattice);
  memoryManager::deallocate(output_lattice);
  memoryManager::deallocate(lattice_ref);

  std::cout << "\n DONE!...\n";
  return 0;
}

//
// Print Lattice
//
void printLattice(int* lattice, int totCellsInRow, int totCellsInCol)
{
  std::cout << std::endl;
  for (int row = 0; row < totCellsInRow; ++row) {
    for (int col = 0; col < totCellsInCol; ++col) {

      const int id = col + totCellsInCol * row;
      std::cout << lattice[id] << " ";
    }
    std::cout << " " << std::endl;
  }
  std::cout << std::endl;
}

//
// Check Result
//
void checkResult(int* compLattice, int* refLattice, int totCells)
{

  bool pass = true;

  for (int i = 0; i < totCells; ++i) {
    if (compLattice[i] != refLattice[i]) pass = false;
  }

  if (pass) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
}
