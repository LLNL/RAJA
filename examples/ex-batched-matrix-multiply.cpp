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
#include "RAJA/util/Timer.hpp"


#include "memoryManager.hpp"

/*
 *  Batched Matrix Multiply Example
 *
 *  This example carries out batched matrix multiplication
 *  for matrices of dimension 3 x 3 using two different
 *  data layouts.
 *
 *  Matrices are stored in arrays A, and B. Results
 *  are stored in a third array, C.
 *  We introduce the notation A^{e}_rc
 *  to correspond to the matrix entry in the row, r,
 *  column, c, of matrix, e. Below we describe the two
 *  layouts for the case of two (N=2) 3 x 3 matrices.
 *
 *  Layout 1:
 *  Matrix entries are grouped together so that each
 *  matrix is in a row major ordering.
 *  i.e. A = [A^{0}_{00}, A^{0}_{01}, A^{0}_{02},
 *            A^{0}_{10}, A^{0}_{11}, A^{0}_{12},
 *            A^{0}_{20}, A^{0}_{21}, A^{0}_{22},
 *            A^{1}_{00}, A^{1}_{01}, A^{1}_{02},
 *            A^{1}_{10}, A^{1}_{11}, A^{1}_{12},
 *            A^{1}_{20}, A^{1}_{21}, A^{1}_{22}];
 *
 *  Layout 2:
 *  Matrix entries are first ordered by matrix number,
 *  then by column number, and finally by row number.
 *  i.e. A = [A^{0}_{00}, A^{1}_{00}, A^{0}_{01},
 *            A^{1}_{01}, A^{0}_{02}, A^{1}_{02},
 *            A^{0}_{10}, A^{1}_{10}, A^{0}_{11},
 *            A^{1}_{11}, A^{0}_{12}, A^{1}_{12},
 *            A^{0}_{20}, A^{1}_{20}, A^{0}_{21},
 *            A^{1}_{21}, A^{0}_{22}, A^{1}_{22}];
 *
 * The extension to N > 2 matrices follows by direct
 * extension. By exploring different data layouts,
 * we can assess which performs best under a given
 * execution policy and architure.
 *
 *  RAJA features shown:
 *    -  RAJA View
 *    -  RAJA make_permuted_layout
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

/*
  CUDA_BLOCK_SIZE - specifies the number of threads in a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 256;
#endif

//
// By default a RAJA::Index_type
// is a long int
//
using RAJA::Index_type;

//
// Function for comparing outputs
//
template <typename T>
void checkResult(T C, Index_type noMat, int nRows, int nCols);

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA batched matrix multiplication example...\n";

  // Dimensions of matrices
  const int NCOLS = 3;
  const int NROWS = 3;

  // Number of matrices
  const Index_type N = 8000000;

  // Number of iterations
  const int NITER = 20;

  std::cout << "Number of matrices to be multiplied: " << N << " \n";

  //
  // Initialize a RAJA timer object
  // and variable to store minimum run time
  //
  auto timer = RAJA::Timer();
  double minRun;

  //
  // Allocate space for data in layout 1
  //
  double *A = memoryManager::allocate<double>(NCOLS * NROWS * N);
  double *B = memoryManager::allocate<double>(NCOLS * NROWS * N);
  double *C = memoryManager::allocate<double>(NCOLS * NROWS * N);

  //
  // Layout 1
  //
  // make_permuted_layout takes the number of entries in each dimension and a
  // templated array indicating slowest to fastest stride. Dimensions are stored
  // in an array object. Here double braces are used to initialize the array and its
  // subobjects (number of entries in each component).
  // The layout generates an indexing equivalent to
  // A(e,r,c) A[c + NCOLS*(r + NROWS*e)]
  auto layout =
      RAJA::make_permuted_layout({{N, NROWS, NCOLS}},
                                 RAJA::as_array<RAJA::Perm<0, 1, 2>>::get());

  // RAJA::Layout is templated on dimension, argument type, and index with unit
  // stride (in this case argument 2 has unit stride)
  RAJA::View<double, RAJA::Layout<3, Index_type, 2>> Aview(A, layout);
  RAJA::View<double, RAJA::Layout<3, Index_type, 2>> Bview(B, layout);
  RAJA::View<double, RAJA::Layout<3, Index_type, 2>> Cview(C, layout);

  //
  // Allocate space for data in layout 2
  //
  double *A2 = memoryManager::allocate<double>(NCOLS * NROWS * N);
  double *B2 = memoryManager::allocate<double>(NCOLS * NROWS * N);
  double *C2 = memoryManager::allocate<double>(NCOLS * NROWS * N);

  //
  // Permuted layout - equivalent to indexing via
  // A[e + N*(r + NROWS*c)]
  auto layout2 =
      RAJA::make_permuted_layout({{N, NROWS, NCOLS}},
                                 RAJA::as_array<RAJA::Perm<1, 2, 0>>::get());
  RAJA::View<double, RAJA::Layout<3, Index_type, 0>> Aview2(A2, layout2);
  RAJA::View<double, RAJA::Layout<3, Index_type, 0>> Bview2(B2, layout2);
  RAJA::View<double, RAJA::Layout<3, Index_type, 0>> Cview2(C2, layout2);

//
// Initialize data
//
//#if defined(RAJA_ENABLE_OPENMP)
  using INIT_POL = RAJA::omp_parallel_for_exec;
  //#else
  //  using INIT_POL = RAJA::loop_exec;
  //#endif

  RAJA::forall<INIT_POL>(RAJA::RangeSegment(0, N), [=](Index_type e) {
    for (Index_type row = 0; row < NROWS; ++row) {
      for (Index_type col = 0; col < NCOLS; ++col) {
        Aview(e, row, col) = row;
        Bview(e, row, col) = col;
        Cview(e, row, col) = 0;

        Aview2(e, row, col) = row;
        Bview2(e, row, col) = col;
        Cview2(e, row, col) = 0;
      }
    }
  });

//-------------------------------------------
// Matrix multiply with layout 1 on the CPU using omp_parallel_for_exec
//

#if defined(RAJA_ENABLE_OPENMP)
  minRun = std::numeric_limits<double>::max();
  for (int i = 0; i < NITER; ++i) {

    timer.start();
    RAJA::forall<RAJA::omp_parallel_for_exec>(
        RAJA::RangeSegment(0, N), [=](Index_type e) {

          Cview(e, 0, 0) = Aview(e, 0, 0) * Bview(e, 0, 0)
                           + Aview(e, 0, 1) * Bview(e, 1, 0)
                           + Aview(e, 0, 2) * Bview(e, 2, 0);
          Cview(e, 0, 1) = Aview(e, 0, 0) * Bview(e, 0, 1)
                           + Aview(e, 0, 1) * Bview(e, 1, 1)
                           + Aview(e, 0, 2) * Bview(e, 2, 1);
          Cview(e, 0, 2) = Aview(e, 0, 0) * Bview(e, 0, 2)
                           + Aview(e, 0, 1) * Bview(e, 1, 2)
                           + Aview(e, 0, 2) * Bview(e, 2, 2);

          Cview(e, 1, 0) = Aview(e, 1, 0) * Bview(e, 0, 0)
                           + Aview(e, 1, 1) * Bview(e, 1, 0)
                           + Aview(e, 1, 2) * Bview(e, 2, 0);
          Cview(e, 1, 1) = Aview(e, 1, 0) * Bview(e, 0, 1)
                           + Aview(e, 1, 1) * Bview(e, 1, 1)
                           + Aview(e, 1, 2) * Bview(e, 2, 1);
          Cview(e, 1, 2) = Aview(e, 1, 0) * Bview(e, 0, 2)
                           + Aview(e, 1, 1) * Bview(e, 1, 2)
                           + Aview(e, 1, 2) * Bview(e, 2, 2);

          Cview(e, 2, 0) = Aview(e, 2, 0) * Bview(e, 0, 0)
                           + Aview(e, 2, 1) * Bview(e, 1, 0)
                           + Aview(e, 2, 2) * Bview(e, 2, 0);
          Cview(e, 2, 1) = Aview(e, 2, 0) * Bview(e, 0, 1)
                           + Aview(e, 2, 1) * Bview(e, 1, 1)
                           + Aview(e, 2, 2) * Bview(e, 2, 1);
          Cview(e, 2, 2) = Aview(e, 2, 0) * Bview(e, 0, 2)
                           + Aview(e, 2, 1) * Bview(e, 1, 2)
                           + Aview(e, 2, 2) * Bview(e, 2, 2);

        });
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }
  std::cout << "Performing batched matrix multiplication with layout 1 using "
               "RAJA::omp_parallel_for_exec ... "
               "run time : "
            << minRun << " seconds" << std::endl;
  checkResult(Cview, N, NROWS, NCOLS);
  //-------------------------------------------

  //-------------------------------------------
  // Matrix multiply with layout 2 on the CPU using omp_parallel_for_exec
  //
  minRun = std::numeric_limits<double>::max();
  for (int i = 0; i < NITER; ++i) {

    timer.start();
    RAJA::forall<RAJA::omp_parallel_for_exec>(
        RAJA::RangeSegment(0, N), [=](Index_type e) {

          Cview2(e, 0, 0) = Aview2(e, 0, 0) * Bview2(e, 0, 0)
                            + Aview2(e, 0, 1) * Bview2(e, 1, 0)
                            + Aview2(e, 0, 2) * Bview2(e, 2, 0);
          Cview2(e, 0, 1) = Aview2(e, 0, 0) * Bview2(e, 0, 1)
                            + Aview2(e, 0, 1) * Bview2(e, 1, 1)
                            + Aview2(e, 0, 2) * Bview2(e, 2, 1);
          Cview2(e, 0, 2) = Aview2(e, 0, 0) * Bview2(e, 0, 2)
                            + Aview2(e, 0, 1) * Bview2(e, 1, 2)
                            + Aview2(e, 0, 2) * Bview2(e, 2, 2);

          Cview2(e, 1, 0) = Aview2(e, 1, 0) * Bview2(e, 0, 0)
                            + Aview2(e, 1, 1) * Bview2(e, 1, 0)
                            + Aview2(e, 1, 2) * Bview2(e, 2, 0);
          Cview2(e, 1, 1) = Aview2(e, 1, 0) * Bview2(e, 0, 1)
                            + Aview2(e, 1, 1) * Bview2(e, 1, 1)
                            + Aview2(e, 1, 2) * Bview2(e, 2, 1);
          Cview2(e, 1, 2) = Aview2(e, 1, 0) * Bview2(e, 0, 2)
                            + Aview2(e, 1, 1) * Bview2(e, 1, 2)
                            + Aview2(e, 1, 2) * Bview2(e, 2, 2);

          Cview2(e, 2, 0) = Aview2(e, 2, 0) * Bview2(e, 0, 0)
                            + Aview2(e, 2, 1) * Bview2(e, 1, 0)
                            + Aview2(e, 2, 2) * Bview2(e, 2, 0);
          Cview2(e, 2, 1) = Aview2(e, 2, 0) * Bview2(e, 0, 1)
                            + Aview2(e, 2, 1) * Bview2(e, 1, 1)
                            + Aview2(e, 2, 2) * Bview2(e, 2, 1);
          Cview2(e, 2, 2) = Aview2(e, 2, 0) * Bview2(e, 0, 2)
                            + Aview2(e, 2, 1) * Bview2(e, 1, 2)
                            + Aview2(e, 2, 2) * Bview2(e, 2, 2);

        });
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }
  std::cout << "Performing batched matrix multiplication with layout 2 using "
               "RAJA::omp_parallel_for_exec ... "
               "run time : "
            << minRun << " seconds" << std::endl;
  checkResult(Cview2, N, NROWS, NCOLS);
//---------------------------------------------
#endif

  //-------------------------------------------
  // Matrix multiply with layout 1 on the CPU with loop_exec policy
  //

  minRun = std::numeric_limits<double>::max();
  for (int i = 0; i < NITER; ++i) {

    timer.start();
    RAJA::forall<RAJA::loop_exec>(RAJA::RangeSegment(0, N), [=](Index_type e) {

      Cview(e, 0, 0) = Aview(e, 0, 0) * Bview(e, 0, 0)
                       + Aview(e, 0, 1) * Bview(e, 1, 0)
                       + Aview(e, 0, 2) * Bview(e, 2, 0);
      Cview(e, 0, 1) = Aview(e, 0, 0) * Bview(e, 0, 1)
                       + Aview(e, 0, 1) * Bview(e, 1, 1)
                       + Aview(e, 0, 2) * Bview(e, 2, 1);
      Cview(e, 0, 2) = Aview(e, 0, 0) * Bview(e, 0, 2)
                       + Aview(e, 0, 1) * Bview(e, 1, 2)
                       + Aview(e, 0, 2) * Bview(e, 2, 2);

      Cview(e, 1, 0) = Aview(e, 1, 0) * Bview(e, 0, 0)
                       + Aview(e, 1, 1) * Bview(e, 1, 0)
                       + Aview(e, 1, 2) * Bview(e, 2, 0);
      Cview(e, 1, 1) = Aview(e, 1, 0) * Bview(e, 0, 1)
                       + Aview(e, 1, 1) * Bview(e, 1, 1)
                       + Aview(e, 1, 2) * Bview(e, 2, 1);
      Cview(e, 1, 2) = Aview(e, 1, 0) * Bview(e, 0, 2)
                       + Aview(e, 1, 1) * Bview(e, 1, 2)
                       + Aview(e, 1, 2) * Bview(e, 2, 2);

      Cview(e, 2, 0) = Aview(e, 2, 0) * Bview(e, 0, 0)
                       + Aview(e, 2, 1) * Bview(e, 1, 0)
                       + Aview(e, 2, 2) * Bview(e, 2, 0);
      Cview(e, 2, 1) = Aview(e, 2, 0) * Bview(e, 0, 1)
                       + Aview(e, 2, 1) * Bview(e, 1, 1)
                       + Aview(e, 2, 2) * Bview(e, 2, 1);
      Cview(e, 2, 2) = Aview(e, 2, 0) * Bview(e, 0, 2)
                       + Aview(e, 2, 1) * Bview(e, 1, 2)
                       + Aview(e, 2, 2) * Bview(e, 2, 2);
    });
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }
  std::cout << "Performing batched matrix multiplication with layout 1 using "
               "RAJA::loop_exec ... "
               "run time : "
            << minRun << " seconds" << std::endl;
  checkResult(Cview, N, NROWS, NCOLS);
  //-------------------------------------------

  //-------------------------------------------
  // Matrix multiply with layout 2 on the CPU
  //
  minRun = std::numeric_limits<double>::max();
  for (int i = 0; i < NITER; ++i) {

    timer.start();
    RAJA::forall<RAJA::loop_exec>(RAJA::RangeSegment(0, N), [=](Index_type e) {

      Cview2(e, 0, 0) = Aview2(e, 0, 0) * Bview2(e, 0, 0)
                        + Aview2(e, 0, 1) * Bview2(e, 1, 0)
                        + Aview2(e, 0, 2) * Bview2(e, 2, 0);
      Cview2(e, 0, 1) = Aview2(e, 0, 0) * Bview2(e, 0, 1)
                        + Aview2(e, 0, 1) * Bview2(e, 1, 1)
                        + Aview2(e, 0, 2) * Bview2(e, 2, 1);
      Cview2(e, 0, 2) = Aview2(e, 0, 0) * Bview2(e, 0, 2)
                        + Aview2(e, 0, 1) * Bview2(e, 1, 2)
                        + Aview2(e, 0, 2) * Bview2(e, 2, 2);

      Cview2(e, 1, 0) = Aview2(e, 1, 0) * Bview2(e, 0, 0)
                        + Aview2(e, 1, 1) * Bview2(e, 1, 0)
                        + Aview2(e, 1, 2) * Bview2(e, 2, 0);
      Cview2(e, 1, 1) = Aview2(e, 1, 0) * Bview2(e, 0, 1)
                        + Aview2(e, 1, 1) * Bview2(e, 1, 1)
                        + Aview2(e, 1, 2) * Bview2(e, 2, 1);
      Cview2(e, 1, 2) = Aview2(e, 1, 0) * Bview2(e, 0, 2)
                        + Aview2(e, 1, 1) * Bview2(e, 1, 2)
                        + Aview2(e, 1, 2) * Bview2(e, 2, 2);

      Cview2(e, 2, 0) = Aview2(e, 2, 0) * Bview2(e, 0, 0)
                        + Aview2(e, 2, 1) * Bview2(e, 1, 0)
                        + Aview2(e, 2, 2) * Bview2(e, 2, 0);
      Cview2(e, 2, 1) = Aview2(e, 2, 0) * Bview2(e, 0, 1)
                        + Aview2(e, 2, 1) * Bview2(e, 1, 1)
                        + Aview2(e, 2, 2) * Bview2(e, 2, 1);
      Cview2(e, 2, 2) = Aview2(e, 2, 0) * Bview2(e, 0, 2)
                        + Aview2(e, 2, 1) * Bview2(e, 1, 2)
                        + Aview2(e, 2, 2) * Bview2(e, 2, 2);

    });
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }
  std::cout << "Performing batched matrix multiplication with layout 2 using "
               "RAJA::loop_exec ... "
               "run time : "
            << minRun << " seconds" << std::endl;
  checkResult(Cview2, N, NROWS, NCOLS);
//---------------------------------------------

#if defined(RAJA_ENABLE_CUDA)
  //-------------------------------------------
  // Matrix multiply with layout 1 on the GPU
  //
  minRun = std::numeric_limits<double>::max();
  for (int i = 0; i < NITER; ++i) {

    timer.start();
    RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(
        RAJA::RangeSegment(0, N), [=] RAJA_DEVICE(Index_type e) {

          Cview(e, 0, 0) = Aview(e, 0, 0) * Bview(e, 0, 0)
                           + Aview(e, 0, 1) * Bview(e, 1, 0)
                           + Aview(e, 0, 2) * Bview(e, 2, 0);
          Cview(e, 0, 1) = Aview(e, 0, 0) * Bview(e, 0, 1)
                           + Aview(e, 0, 1) * Bview(e, 1, 1)
                           + Aview(e, 0, 2) * Bview(e, 2, 1);
          Cview(e, 0, 2) = Aview(e, 0, 0) * Bview(e, 0, 2)
                           + Aview(e, 0, 1) * Bview(e, 1, 2)
                           + Aview(e, 0, 2) * Bview(e, 2, 2);

          Cview(e, 1, 0) = Aview(e, 1, 0) * Bview(e, 0, 0)
                           + Aview(e, 1, 1) * Bview(e, 1, 0)
                           + Aview(e, 1, 2) * Bview(e, 2, 0);
          Cview(e, 1, 1) = Aview(e, 1, 0) * Bview(e, 0, 1)
                           + Aview(e, 1, 1) * Bview(e, 1, 1)
                           + Aview(e, 1, 2) * Bview(e, 2, 1);
          Cview(e, 1, 2) = Aview(e, 1, 0) * Bview(e, 0, 2)
                           + Aview(e, 1, 1) * Bview(e, 1, 2)
                           + Aview(e, 1, 2) * Bview(e, 2, 2);

          Cview(e, 2, 0) = Aview(e, 2, 0) * Bview(e, 0, 0)
                           + Aview(e, 2, 1) * Bview(e, 1, 0)
                           + Aview(e, 2, 2) * Bview(e, 2, 0);
          Cview(e, 2, 1) = Aview(e, 2, 0) * Bview(e, 0, 1)
                           + Aview(e, 2, 1) * Bview(e, 1, 1)
                           + Aview(e, 2, 2) * Bview(e, 2, 1);
          Cview(e, 2, 2) = Aview(e, 2, 0) * Bview(e, 0, 2)
                           + Aview(e, 2, 1) * Bview(e, 1, 2)
                           + Aview(e, 2, 2) * Bview(e, 2, 2);

        });
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }
  std::cout << "Matrix Multiplication with layout 1 on GPU with "
               "RAJA::cuda_exec ... "
            << minRun << " seconds" << std::endl;
  checkResult(Cview, N, NROWS, NCOLS);
  //---------------------------------------------

  //-------------------------------------------
  // Matrix multiply with layout 2 on the GPU
  //
  minRun = std::numeric_limits<double>::max();
  for (int i = 0; i < NITER; ++i) {

    timer.start();
    RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(
        RAJA::RangeSegment(0, N), [=] RAJA_DEVICE(Index_type e) {

          Cview2(e, 0, 0) = Aview2(e, 0, 0) * Bview2(e, 0, 0)
                            + Aview2(e, 0, 1) * Bview2(e, 1, 0)
                            + Aview2(e, 0, 2) * Bview2(e, 2, 0);
          Cview2(e, 0, 1) = Aview2(e, 0, 0) * Bview2(e, 0, 1)
                            + Aview2(e, 0, 1) * Bview2(e, 1, 1)
                            + Aview2(e, 0, 2) * Bview2(e, 2, 1);
          Cview2(e, 0, 2) = Aview2(e, 0, 0) * Bview2(e, 0, 2)
                            + Aview2(e, 0, 1) * Bview2(e, 1, 2)
                            + Aview2(e, 0, 2) * Bview2(e, 2, 2);

          Cview2(e, 1, 0) = Aview2(e, 1, 0) * Bview2(e, 0, 0)
                            + Aview2(e, 1, 1) * Bview2(e, 1, 0)
                            + Aview2(e, 1, 2) * Bview2(e, 2, 0);
          Cview2(e, 1, 1) = Aview2(e, 1, 0) * Bview2(e, 0, 1)
                            + Aview2(e, 1, 1) * Bview2(e, 1, 1)
                            + Aview2(e, 1, 2) * Bview2(e, 2, 1);
          Cview2(e, 1, 2) = Aview2(e, 1, 0) * Bview2(e, 0, 2)
                            + Aview2(e, 1, 1) * Bview2(e, 1, 2)
                            + Aview2(e, 1, 2) * Bview2(e, 2, 2);

          Cview2(e, 2, 0) = Aview2(e, 2, 0) * Bview2(e, 0, 0)
                            + Aview2(e, 2, 1) * Bview2(e, 1, 0)
                            + Aview2(e, 2, 2) * Bview2(e, 2, 0);
          Cview2(e, 2, 1) = Aview2(e, 2, 0) * Bview2(e, 0, 1)
                            + Aview2(e, 2, 1) * Bview2(e, 1, 1)
                            + Aview2(e, 2, 2) * Bview2(e, 2, 1);
          Cview2(e, 2, 2) = Aview2(e, 2, 0) * Bview2(e, 0, 2)
                            + Aview2(e, 2, 1) * Bview2(e, 1, 2)
                            + Aview2(e, 2, 2) * Bview2(e, 2, 2);

        });
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }
  std::cout << "Matrix Multiplication with layout 2 on GPU with "
               "RAJA::cuda_exec ... "
            << minRun << " seconds" << std::endl;
  checkResult(Cview, N, NROWS, NCOLS);
//---------------------------------------------
#endif
  //
  // Clean up.
  //
  memoryManager::deallocate(A);
  memoryManager::deallocate(B);
  memoryManager::deallocate(C);
  memoryManager::deallocate(C2);

  std::cout << "\n DONE!...\n";
  return 0;
}

//
// check result
//
template <typename T>
void checkResult(T C, Index_type noMat, int nRows, int nCols)
{

  bool status = true;
  for (int e = 0; e < noMat; ++e) {
    for (int row = 0; row < nRows; ++row) {
      for (int col = 0; col < nCols; ++col) {
        if (std::abs(C(e, row, col) - row * col * nCols) > 10e-12) {
          status = false;
        }
      }
    }
  }

  if (status == false) {
    std::cout << "Batched Matrix Multiply - fail" << std::endl;
  } else {
    std::cout << "Batched Matrix Multiply - pass" << std::endl;
  }
}
