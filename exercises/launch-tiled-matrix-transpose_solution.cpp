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

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Tiled Matrix Transpose Example
 *
 *  In this example, an input matrix A of dimension N_r x N_c is
 *  transposed and returned as a second matrix At.
 *
 *  This operation is carried out using a tiling algorithm.
 *  The algorithm iterates over tiles of the matrix A and
 *  performs a transpose copy without explicitly storing the tile.
 *
 *  The algorithm is expressed as a collection of ``outer``
 *  and ``inner`` for loops. Iterations of the inner loop will
 *  tranpose tile entries; while outer loops will iterate over
 *  the number of tiles needed to carryout the transpose.
 *  We do not assume that tiles divide the number of rows and
 *  and columns of the matrix.
 *
 *  RAJA features shown:
 *    - Basic usage of 'RAJA::kernel' abstractions for nested loops
 *    - Tiling statement
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

//
// Define dimensionality of matrices
//
const int DIM = 2;

//
// Function for checking results
//
template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N_r, int N_c);

//
// Function for printing results
//
template <typename T>
void printResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N_r, int N_c);


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA tiled matrix transpose example...\n";

  //
  // Define num rows/cols in matrix, tile dimensions, and number of tiles.
  //
  // _tiled_mattranspose_dims_start
  const int N_r = 56;
  const int N_c = 75;

  const int TILE_DIM = 16;

  const int outer_Dimc = (N_c - 1) / TILE_DIM + 1;
  const int outer_Dimr = (N_r - 1) / TILE_DIM + 1;
  // _tiled_mattranspose_dims_end

  //
  // Allocate matrix data
  //
  int *A = memoryManager::allocate<int>(N_r * N_c);
  int *At = memoryManager::allocate<int>(N_r * N_c);

  //
  // In the following implementations of tiled matrix transpose, we
  // use RAJA 'View' objects to access the matrix data. A RAJA view
  // holds a pointer to a data array and enables multi-dimensional indexing
  // into the data.
  //
  // _tiled_mattranspose_views_start
  RAJA::View<int, RAJA::Layout<DIM>> Aview(A, N_r, N_c);
  RAJA::View<int, RAJA::Layout<DIM>> Atview(At, N_c, N_r);
  // _tiled_mattranspose_views_end

  //
  // Initialize matrix data
  //
  for (int row = 0; row < N_r; ++row) {
    for (int col = 0; col < N_c; ++col) {
      Aview(row, col) = col;
    }
  }
  //printResult<int>(Aview, N_r, N_c);


  //----------------------------------------------------------------------------//
  std::cout << "\n Running C-version of tiled matrix transpose...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  // _cstyle_tiled_mattranspose_start
  //
  // (0) Outer loops to iterate over tiles
  //
  for (int by = 0; by < outer_Dimr; ++by) {
    for (int bx = 0; bx < outer_Dimc; ++bx) {
      //
      // (1) Loops to iterate over tile entries
      //
      for (int ty = 0; ty < TILE_DIM; ++ty) {
        for (int tx = 0; tx < TILE_DIM; ++tx) {

          int col = bx * TILE_DIM + tx;  // Matrix column index
          int row = by * TILE_DIM + ty;  // Matrix row index

          // Bounds check
          if (row < N_r && col < N_c) {
            Atview(col, row) = Aview(row, col);
          }
        }
      }

    }
  }
  // _cstyle_tiled_mattranspose_end

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);
  //----------------------------------------------------------------------------//

  //
  // The following RAJA variants use the RAJA::kernel method to carryout the
  // transpose.
  //
  // Here, we define RAJA range segments to establish the iteration spaces.
  // Further partioning of the iteration space is carried out in the
  // tile_fixed statements. Iterations inside a RAJA loop is given by their
  // global iteration number.
  //
  RAJA::RangeSegment row_Range(0, N_r);
  RAJA::RangeSegment col_Range(0, N_c);

  //----------------------------------------------------------------------------//
  std::cout << "\n Running sequential tiled matrix transpose ...\n";
  std::memset(At, 0, N_r * N_c * sizeof(int));

  //
  // The following policy carries out the transpose
  // using sequential loops. The template parameter inside
  // tile_fixed corresponds to the dimension size of the tile.
  //
  // _raja_tiled_mattranspose_start
  using loop_pol_1 = RAJA::expt::LoopPolicy<RAJA::loop_exec>;
  using launch_policy_1 = RAJA::expt::LaunchPolicy<RAJA::expt::seq_launch_t>;

  RAJA::expt::launch<launch_policy_1>
    (RAJA::expt::Grid(), //Grid may be empty when running on the cpu
    [=] RAJA_HOST_DEVICE (RAJA::expt::LaunchContext ctx) {

      RAJA::expt::tile<loop_pol_1>
        (ctx, TILE_DIM, row_Range, [&] (RAJA::RangeSegment const &row_tile) {

        RAJA::expt::tile<loop_pol_1>
          (ctx, TILE_DIM, col_Range, [&] (RAJA::RangeSegment const &col_tile) {

            RAJA::expt::loop<loop_pol_1>(ctx, row_tile, [&] (int row) {
                RAJA::expt::loop<loop_pol_1>(ctx, col_tile, [&] (int col) {

                    Atview(col, row) = Aview(row, col);

                  });
              });

          });
        });

    });
  // _raja_tiled_mattranspose_end

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);

//----------------------------------------------------------------------------//
#if defined(RAJA_ENABLE_OPENMP)
  std::cout << "\n Running openmp tiled matrix transpose -  parallel top inner loop...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  //
  // This policy loops over tiles sequentially while exposing parallelism on
  // one of the inner loops.
  //
  using omp_for_pol_2 = RAJA::expt::LoopPolicy<RAJA::omp_for_exec>;
  using loop_pol_2 = RAJA::expt::LoopPolicy<RAJA::loop_exec>;
  using launch_policy_2 = RAJA::expt::LaunchPolicy<RAJA::expt::omp_launch_t>;

  RAJA::expt::launch<launch_policy_2>
    (RAJA::expt::Grid(), //Grid may be empty when running on the cpu
    [=] RAJA_HOST_DEVICE (RAJA::expt::LaunchContext ctx) {

      RAJA::expt::tile<omp_for_pol_2>
        (ctx, TILE_DIM, row_Range, [&] (RAJA::RangeSegment const &row_tile) {

        RAJA::expt::tile<loop_pol_2>
          (ctx, TILE_DIM, col_Range, [&] (RAJA::RangeSegment const &col_tile) {

            RAJA::expt::loop<loop_pol_2>(ctx, row_tile, [&] (int row) {
                RAJA::expt::loop<loop_pol_2>(ctx, col_tile, [&] (int col) {

                    Atview(col, row) = Aview(row, col);

                  });
              });

          });
        });

    });

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);

#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)
  std::cout << "\n Running cuda tiled matrix transpose ...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  constexpr int c_block_sz = TILE_DIM;
  constexpr int r_block_sz = TILE_DIM;
  const int n_blocks_c = RAJA_DIVIDE_CEILING_INT(N_c, c_block_sz);
  const int n_blocks_r = RAJA_DIVIDE_CEILING_INT(N_r, r_block_sz);

  using cuda_teams_y = RAJA::expt::LoopPolicy<RAJA::cuda_block_y_direct>;
  using cuda_teams_x = RAJA::expt::LoopPolicy<RAJA::cuda_block_x_direct>;

  using cuda_threads_y = RAJA::expt::LoopPolicy<RAJA::cuda_thread_y_direct>;
  using cuda_threads_x = RAJA::expt::LoopPolicy<RAJA::cuda_thread_x_direct>;

  const bool cuda_async = false;
  using cuda_launch_policy = RAJA::expt::LaunchPolicy<RAJA::expt::cuda_launch_t<cuda_async>>;

  RAJA::expt::launch<cuda_launch_policy>
    (RAJA::expt::Grid(RAJA::expt::Teams(n_blocks_c, n_blocks_r),
                      RAJA::expt::Threads(c_block_sz, r_block_sz)),
    [=] RAJA_HOST_DEVICE (RAJA::expt::LaunchContext ctx) {

      RAJA::expt::tile<cuda_teams_y>
        (ctx, TILE_DIM, row_Range, [&] (RAJA::RangeSegment const &row_tile) {

        RAJA::expt::tile<cuda_teams_x>
          (ctx, TILE_DIM, col_Range, [&] (RAJA::RangeSegment const &col_tile) {

            RAJA::expt::loop<cuda_threads_y>(ctx, row_tile, [&] (int row) {
                RAJA::expt::loop<cuda_threads_x>(ctx, col_tile, [&] (int col) {

                    Atview(col, row) = Aview(row, col);

                  });
              });

          });
        });

    });

  checkResult<int>(Atview, N_c, N_r);
  //printResult<int>(Atview, N_c, N_r);
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)
  std::cout << "\n Running hip tiled matrix transpose ...\n";

  int* d_A  = memoryManager::allocate_gpu<int>(N_r * N_c);
  int* d_At = memoryManager::allocate_gpu<int>(N_r * N_c);

  RAJA::View<int, RAJA::Layout<DIM>> d_Aview(d_A, N_r, N_c);
  RAJA::View<int, RAJA::Layout<DIM>> d_Atview(d_At, N_c, N_r);

  std::memset(At, 0, N_r * N_c * sizeof(int));
  hipErrchk(hipMemcpy( d_A, A, N_r * N_c * sizeof(int), hipMemcpyHostToDevice ));
  hipErrchk(hipMemcpy( d_At, At, N_r * N_c * sizeof(int), hipMemcpyHostToDevice ));

  constexpr int c_block_sz = TILE_DIM;
  constexpr int r_block_sz = TILE_DIM;
  const int n_blocks_c = RAJA_DIVIDE_CEILING_INT(N_c, c_block_sz);
  const int n_blocks_r = RAJA_DIVIDE_CEILING_INT(N_r, r_block_sz);

  using hip_teams_y = RAJA::expt::LoopPolicy<RAJA::hip_block_y_direct>;
  using hip_teams_x = RAJA::expt::LoopPolicy<RAJA::hip_block_x_direct>;

  using hip_threads_y = RAJA::expt::LoopPolicy<RAJA::hip_thread_y_direct>;
  using hip_threads_x = RAJA::expt::LoopPolicy<RAJA::hip_thread_x_direct>;

  const bool hip_async = false;
  using hip_launch_policy = RAJA::expt::LaunchPolicy<RAJA::expt::hip_launch_t<hip_async>>;

  RAJA::expt::launch<hip_launch_policy>
    (RAJA::expt::Grid(RAJA::expt::Teams(n_blocks_c, n_blocks_r),
                      RAJA::expt::Threads(c_block_sz, r_block_sz)),
    [=] RAJA_HOST_DEVICE (RAJA::expt::LaunchContext ctx) {

      RAJA::expt::tile<hip_teams_y>
        (ctx, TILE_DIM, row_Range, [&] (RAJA::RangeSegment const &row_tile) {

        RAJA::expt::tile<hip_teams_x>
          (ctx, TILE_DIM, col_Range, [&] (RAJA::RangeSegment const &col_tile) {

            RAJA::expt::loop<hip_threads_y>(ctx, row_tile, [&] (int row) {
                RAJA::expt::loop<hip_threads_x>(ctx, col_tile, [&] (int col) {

                    Atview(col, row) = Aview(row, col);

                  });
              });

          });
        });

    });

  hipErrchk(hipMemcpy( At, d_At, N_r * N_c * sizeof(int), hipMemcpyDeviceToHost ));
  checkResult<int>(Atview, N_c, N_r);
  //printResult<int>(Atview, N_c, N_r);
#endif

  //----------------------------------------------------------------------------//


  //
  // Clean up.
  //
  memoryManager::deallocate(A);
  memoryManager::deallocate(At);

  std::cout << "\n DONE!...\n";

  return 0;
}

//
// Function to check result and report P/F.
//
template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N_r, int N_c)
{
  bool match = true;
  for (int row = 0; row < N_r; ++row) {
    for (int col = 0; col < N_c; ++col) {
      if (Atview(row, col) != row) {
        match = false;
      }
    }
  }
  if (match) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
};

//
// Function to print result.
//
template <typename T>
void printResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N_r, int N_c)
{
  std::cout << std::endl;
  for (int row = 0; row < N_r; ++row) {
    for (int col = 0; col < N_c; ++col) {
      // std::cout << "At(" << row << "," << col << ") = " << Atview(row, col)
      //                << std::endl;
      std::cout<<Atview(row, col)<<" ";
    }
    std::cout << "" << std::endl;
  }
  std::cout << std::endl;
}
