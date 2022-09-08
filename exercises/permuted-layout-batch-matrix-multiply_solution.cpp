//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
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
 *  This example performs batched matrix multiplication
 *  for matrices of dimension 3 x 3 using two different
 *  data layouts.
 *
 *  Matrices are stored in arrays A and B. Results
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
 * execution policy and architecture.
 *
 *  RAJA features shown:
 *    - `forall` loop iteration template method
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

#if defined(RAJA_ENABLE_HIP)
const int HIP_BLOCK_SIZE = 256;
#endif

//
// By default a RAJA::Index_type
// is a long int
//
using RAJA::Index_type;

//
//Function for checking results
//
template <typename T>
void checkResult(T C, Index_type noMat, int nRows, int nCols);

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA batched matrix multiplication example...\n";

// Dimensions of matrices
  const int N_c = 3;
  const int N_r = 3;

// Number of matrices
  const Index_type N = 8000000;

// Number of iterations
  const int NITER = 20;

  std::cout << "\n Number of matrices to be multiplied: " << N << " \n \n";

//
// Initialize a RAJA timer object
// and variable to store minimum run time
//
  auto timer = RAJA::Timer();
  double minRun;

//
// Allocate space for data in layout 1
//
  double *A = memoryManager::allocate<double>(N_c * N_r * N);
  double *B = memoryManager::allocate<double>(N_c * N_r * N);
  double *C = memoryManager::allocate<double>(N_c * N_r * N);

//
// Layout 1
//
// make_permuted_layout takes the number of entries in each dimension and a
// templated array indicating index arguments with slowest to fastest stride.
// Standard C++ arrays are used to hold the number of entries in each component.
// This example uses double braces to initalize the array and its subobjects.
// The layout object will index into the array as the following C macro would
// #define Aview(e, r, c) A[c + N_c*(r + N_r*e)].
//
// RAJA::Layout objects may be templated on dimension, argument type, and 
// index with unit stride. Here, the column index has unit stride (argument 2). 
//
  // _permutedlayout_defviews_start
  std::array<RAJA::idx_t, 3> perm1 {{0, 1, 2}};
  auto layout1 =
      RAJA::make_permuted_layout( {{N, N_r, N_c}}, perm1 );

  RAJA::View<double, RAJA::Layout<3, Index_type, 2>> Aview(A, layout1);
  RAJA::View<double, RAJA::Layout<3, Index_type, 2>> Bview(B, layout1);
  RAJA::View<double, RAJA::Layout<3, Index_type, 2>> Cview(C, layout1);
  // _permutedlayout_defviews_end

//
// Allocate space for data in layout 2
//
  double *A2 = memoryManager::allocate<double>(N_c * N_r * N);
  double *B2 = memoryManager::allocate<double>(N_c * N_r * N);
  double *C2 = memoryManager::allocate<double>(N_c * N_r * N);

//
// Permuted layout - equivalent to indexing using the following macro
// #define Aview2(e, r, c) A2[e + N*(c + N_c*r)]
// In this case the element index has unit stride (argument 0). 
//
  // _permutedlayout_permviews_start
  std::array<RAJA::idx_t, 3> perm2 {{1, 2, 0}};
  auto layout2 =
      RAJA::make_permuted_layout( {{N, N_r, N_c}}, perm2 );

  RAJA::View<double, RAJA::Layout<3, Index_type, 0>> Aview2(A2, layout2);
  RAJA::View<double, RAJA::Layout<3, Index_type, 0>> Bview2(B2, layout2);
  RAJA::View<double, RAJA::Layout<3, Index_type, 0>> Cview2(C2, layout2);
  // _permutedlayout_permviews_end

//
// Initialize data
//
#if defined(RAJA_ENABLE_OPENMP)
  using INIT_POL = RAJA::omp_parallel_for_exec;
#else
  using INIT_POL = RAJA::loop_exec;
#endif

  RAJA::forall<INIT_POL>(RAJA::RangeSegment(0, N), [=](Index_type e) {
    for (Index_type row = 0; row < N_r; ++row) {
      for (Index_type col = 0; col < N_c; ++col) {
        Aview(e, row, col) = row;
        Bview(e, row, col) = col;
        Cview(e, row, col) = 0;

        Aview2(e, row, col) = row;
        Bview2(e, row, col) = col;
        Cview2(e, row, col) = 0;
      }
    }
  });

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::cout << " \n Performing batched matrix multiplication"
            << " with layout 1 (RAJA - omp parallel for) ... " << std::endl;

  minRun = std::numeric_limits<double>::max();
  for (int i = 0; i < NITER; ++i) {

    timer.start();
    // _permutedlayout_batchedmatmult_omp_start
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
    // _permutedlayout_batchedmatmult_omp_end
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }
  
  std::cout<< "\trun time : " << minRun << " seconds" << std::endl;
  checkResult(Cview, N, N_r, N_c);

//----------------------------------------------------------------------------//

  std::cout << " \n Performing batched matrix multiplication"
            << " with layout 2 (RAJA - omp parallel for) ... " << std::endl;

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
  std::cout<< "\trun time : " << minRun << " seconds" << std::endl;
  checkResult(Cview2, N, N_r, N_c);

#endif

//----------------------------------------------------------------------------//

  std::cout << " \n Performing batched matrix multiplication"
            << " with layout 1 (RAJA - sequential) ... " << std::endl;

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
    
  std::cout << "\trun time : " << minRun << " seconds" << std::endl;
  checkResult(Cview, N, N_r, N_c);

//----------------------------------------------------------------------------//

  std::cout << " \n Performing batched matrix multiplication"
            << " with layout 2 (RAJA - sequential) ... " << std::endl;

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
  std::cout<< "\trun time : "<< minRun << " seconds" << std::endl;
  checkResult(Cview2, N, N_r, N_c);

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

  std::cout << " \n Performing batched matrix multiplication"
            << " with layout 1 (RAJA - cuda) ... " << std::endl;

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

  std::cout<< "\trun time: "<< minRun << " seconds" << std::endl;
  checkResult(Cview, N, N_r, N_c);

//----------------------------------------------------------------------------//

  std::cout << " \n Performing batched matrix multiplication"
            << " with layout 2 (RAJA - cuda) ... " << std::endl;

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
  std::cout<< "\trun time : "<< minRun << " seconds" << std::endl;
  checkResult(Cview2, N, N_r, N_c);
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)

  std::cout << " \n Performing batched matrix multiplication"
            << " with layout 1 (RAJA - hip) ... " << std::endl;

  double *d_A = memoryManager::allocate_gpu<double>(N_c * N_r * N);
  double *d_B = memoryManager::allocate_gpu<double>(N_c * N_r * N);
  double *d_C = memoryManager::allocate_gpu<double>(N_c * N_r * N);

  double *d_A2 = memoryManager::allocate_gpu<double>(N_c * N_r * N);
  double *d_B2 = memoryManager::allocate_gpu<double>(N_c * N_r * N);
  double *d_C2 = memoryManager::allocate_gpu<double>(N_c * N_r * N);

  RAJA::View<double, RAJA::Layout<3, Index_type, 2>> d_Aview(d_A, layout1);
  RAJA::View<double, RAJA::Layout<3, Index_type, 2>> d_Bview(d_B, layout1);
  RAJA::View<double, RAJA::Layout<3, Index_type, 2>> d_Cview(d_C, layout1);

  RAJA::View<double, RAJA::Layout<3, Index_type, 0>> d_Aview2(d_A2, layout2);
  RAJA::View<double, RAJA::Layout<3, Index_type, 0>> d_Bview2(d_B2, layout2);
  RAJA::View<double, RAJA::Layout<3, Index_type, 0>> d_Cview2(d_C2, layout2);

  hipErrchk(hipMemcpy( d_A, A, N_c * N_r * N * sizeof(double), hipMemcpyHostToDevice ));
  hipErrchk(hipMemcpy( d_B, B, N_c * N_r * N * sizeof(double), hipMemcpyHostToDevice ));
  hipErrchk(hipMemcpy( d_A2, A2, N_c * N_r * N * sizeof(double), hipMemcpyHostToDevice ));
  hipErrchk(hipMemcpy( d_B2, B2, N_c * N_r * N * sizeof(double), hipMemcpyHostToDevice ));

  minRun = std::numeric_limits<double>::max();
  for (int i = 0; i < NITER; ++i) {

    timer.start();
    RAJA::forall<RAJA::hip_exec<HIP_BLOCK_SIZE>>(
        RAJA::RangeSegment(0, N), [=] RAJA_DEVICE(Index_type e) {

          d_Cview(e, 0, 0) = d_Aview(e, 0, 0) * d_Bview(e, 0, 0)
                           + d_Aview(e, 0, 1) * d_Bview(e, 1, 0)
                           + d_Aview(e, 0, 2) * d_Bview(e, 2, 0);
          d_Cview(e, 0, 1) = d_Aview(e, 0, 0) * d_Bview(e, 0, 1)
                           + d_Aview(e, 0, 1) * d_Bview(e, 1, 1)
                           + d_Aview(e, 0, 2) * d_Bview(e, 2, 1);
          d_Cview(e, 0, 2) = d_Aview(e, 0, 0) * d_Bview(e, 0, 2)
                           + d_Aview(e, 0, 1) * d_Bview(e, 1, 2)
                           + d_Aview(e, 0, 2) * d_Bview(e, 2, 2);

          d_Cview(e, 1, 0) = d_Aview(e, 1, 0) * d_Bview(e, 0, 0)
                           + d_Aview(e, 1, 1) * d_Bview(e, 1, 0)
                           + d_Aview(e, 1, 2) * d_Bview(e, 2, 0);
          d_Cview(e, 1, 1) = d_Aview(e, 1, 0) * d_Bview(e, 0, 1)
                           + d_Aview(e, 1, 1) * d_Bview(e, 1, 1)
                           + d_Aview(e, 1, 2) * d_Bview(e, 2, 1);
          d_Cview(e, 1, 2) = d_Aview(e, 1, 0) * d_Bview(e, 0, 2)
                           + d_Aview(e, 1, 1) * d_Bview(e, 1, 2)
                           + d_Aview(e, 1, 2) * d_Bview(e, 2, 2);

          d_Cview(e, 2, 0) = d_Aview(e, 2, 0) * d_Bview(e, 0, 0)
                           + d_Aview(e, 2, 1) * d_Bview(e, 1, 0)
                           + d_Aview(e, 2, 2) * d_Bview(e, 2, 0);
          d_Cview(e, 2, 1) = d_Aview(e, 2, 0) * d_Bview(e, 0, 1)
                           + d_Aview(e, 2, 1) * d_Bview(e, 1, 1)
                           + d_Aview(e, 2, 2) * d_Bview(e, 2, 1);
          d_Cview(e, 2, 2) = d_Aview(e, 2, 0) * d_Bview(e, 0, 2)
                           + d_Aview(e, 2, 1) * d_Bview(e, 1, 2)
                           + d_Aview(e, 2, 2) * d_Bview(e, 2, 2);

        });
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }

  hipErrchk(hipMemcpy( C, d_C, N_c * N_r * N * sizeof(double), hipMemcpyDeviceToHost ));

  std::cout<< "\trun time: "<< minRun << " seconds" << std::endl;
  checkResult(Cview, N, N_r, N_c);

//----------------------------------------------------------------------------//

  std::cout << " \n Performing batched matrix multiplication"
            << " with layout 2 (RAJA - hip) ... " << std::endl;

  minRun = std::numeric_limits<double>::max();
  for (int i = 0; i < NITER; ++i) {

    timer.start();
    RAJA::forall<RAJA::hip_exec<HIP_BLOCK_SIZE>>(
        RAJA::RangeSegment(0, N), [=] RAJA_DEVICE(Index_type e) {

          d_Cview2(e, 0, 0) = d_Aview2(e, 0, 0) * d_Bview2(e, 0, 0)
                            + d_Aview2(e, 0, 1) * d_Bview2(e, 1, 0)
                            + d_Aview2(e, 0, 2) * d_Bview2(e, 2, 0);
          d_Cview2(e, 0, 1) = d_Aview2(e, 0, 0) * d_Bview2(e, 0, 1)
                            + d_Aview2(e, 0, 1) * d_Bview2(e, 1, 1)
                            + d_Aview2(e, 0, 2) * d_Bview2(e, 2, 1);
          d_Cview2(e, 0, 2) = d_Aview2(e, 0, 0) * d_Bview2(e, 0, 2)
                            + d_Aview2(e, 0, 1) * d_Bview2(e, 1, 2)
                            + d_Aview2(e, 0, 2) * d_Bview2(e, 2, 2);

          d_Cview2(e, 1, 0) = d_Aview2(e, 1, 0) * d_Bview2(e, 0, 0)
                            + d_Aview2(e, 1, 1) * d_Bview2(e, 1, 0)
                            + d_Aview2(e, 1, 2) * d_Bview2(e, 2, 0);
          d_Cview2(e, 1, 1) = d_Aview2(e, 1, 0) * d_Bview2(e, 0, 1)
                            + d_Aview2(e, 1, 1) * d_Bview2(e, 1, 1)
                            + d_Aview2(e, 1, 2) * d_Bview2(e, 2, 1);
          d_Cview2(e, 1, 2) = d_Aview2(e, 1, 0) * d_Bview2(e, 0, 2)
                            + d_Aview2(e, 1, 1) * d_Bview2(e, 1, 2)
                            + d_Aview2(e, 1, 2) * d_Bview2(e, 2, 2);

          d_Cview2(e, 2, 0) = d_Aview2(e, 2, 0) * d_Bview2(e, 0, 0)
                            + d_Aview2(e, 2, 1) * d_Bview2(e, 1, 0)
                            + d_Aview2(e, 2, 2) * d_Bview2(e, 2, 0);
          d_Cview2(e, 2, 1) = d_Aview2(e, 2, 0) * d_Bview2(e, 0, 1)
                            + d_Aview2(e, 2, 1) * d_Bview2(e, 1, 1)
                            + d_Aview2(e, 2, 2) * d_Bview2(e, 2, 1);
          d_Cview2(e, 2, 2) = d_Aview2(e, 2, 0) * d_Bview2(e, 0, 2)
                            + d_Aview2(e, 2, 1) * d_Bview2(e, 1, 2)
                            + d_Aview2(e, 2, 2) * d_Bview2(e, 2, 2);

        });
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }

  hipErrchk(hipMemcpy( C2, d_C2, N_c * N_r * N * sizeof(double), hipMemcpyDeviceToHost ));

  std::cout<< "\trun time : "<< minRun << " seconds" << std::endl;
  checkResult(Cview2, N, N_r, N_c);

  memoryManager::deallocate_gpu(d_A);
  memoryManager::deallocate_gpu(d_B);
  memoryManager::deallocate_gpu(d_C);
  memoryManager::deallocate_gpu(d_A2);
  memoryManager::deallocate_gpu(d_B2);
  memoryManager::deallocate_gpu(d_C2);
#endif

//----------------------------------------------------------------------------//

//
// Clean up.
//
  memoryManager::deallocate(A);
  memoryManager::deallocate(B);
  memoryManager::deallocate(C);
  memoryManager::deallocate(A2);
  memoryManager::deallocate(B2);
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

  if ( status ) {
    std::cout << "\tresult -- PASS\n";
  } else {
    std::cout << "\tresult -- FAIL\n";
  }
}
