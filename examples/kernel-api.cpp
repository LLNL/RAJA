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

using RAJA::statement::Seg;
using RAJA::statement::Param;
using RAJA::statement::OffSet;
using RAJA::statement::OffSetList;

using RAJA::statement::SegList;
using RAJA::statement::ParamList;

/*
 *  Matrix Transpose Example
 *
 *  Compares kernel APIs
 *
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

  std::cout << "\n\nRAJA shared matrix transpose example...\n";

  //
  // Define num rows/cols in matrix
  //
  const int N_r = 267;
  const int N_c = 251;

  //
  // Allocate matrix data
  //
  int *A = memoryManager::allocate<int>(N_r * N_c);
  int *At = memoryManager::allocate<int>(N_r * N_c);

  //
  // In the following implementations of matrix transpose, we
  // use RAJA 'View' objects to access the matrix data. A RAJA view
  // holds a pointer to a data array and enables multi-dimensional indexing
  // into the data.
  //
  RAJA::View<int, RAJA::Layout<DIM>> Aview(A, N_r, N_c);
  RAJA::View<int, RAJA::Layout<DIM>> Atview(At, N_c, N_r);

  //
  // Define TILE dimensions (TILE_DIM x TILE_DIM)
  //
  const int TILE_DIM = 16;

  // Calculate number of tiles (Needed for C++ version)
  const int outer_Dimc = (N_c - 1) / TILE_DIM + 1;
  const int outer_Dimr = (N_r - 1) / TILE_DIM + 1;

  //
  // Initialize matrix data
  //
  for (int row = 0; row < N_r; ++row) {
    for (int col = 0; col < N_c; ++col) {
      Aview(row, col) = col;
    }
  }
  // printResult<int>(Aview, N_r, N_c);

  //----------------------------------------------------------------------------//
  std::cout << "\n Running C-version of shared matrix transpose...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  //
  // (0) Outer loops to iterate over tiles
  //
  for (int by = 0; by < outer_Dimr; ++by) {
    for (int bx = 0; bx < outer_Dimc; ++bx) {

      // Stack-allocated local array for data on a tile
      int Tile[TILE_DIM][TILE_DIM];

      //
      // (1) Inner loops to read input matrix tile data into the array
      //
      //     Note: loops are ordered so that input matrix data access
      //           is stride-1.
      //
      for (int ty = 0; ty < TILE_DIM; ++ty) {
        for (int tx = 0; tx < TILE_DIM; ++tx) {

          int col = bx * TILE_DIM + tx;  // Matrix column index
          int row = by * TILE_DIM + ty;  // Matrix row index

          // Bounds check
          if (row < N_r && col < N_c) {
            Tile[ty][tx] = Aview(row, col);
          }
        }
      }

      //
      // (2) Inner loops to write array data into output array tile
      //
      //     Note: loop order is swapped from above so that output matrix
      //           data access is stride-1.
      //
      for (int tx = 0; tx < TILE_DIM; ++tx) {
        for (int ty = 0; ty < TILE_DIM; ++ty) {

          int col = bx * TILE_DIM + tx;  // Matrix column index
          int row = by * TILE_DIM + ty;  // Matrix row index

          // Bounds check
          if (row < N_r && col < N_c) {
            Atview(col, row) = Tile[ty][tx];
          }
        }
      }

    }
  }
  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);

  //----------------------------------------------------------------------------//

  //
  // The following RAJA variants use the RAJA::Kernel
  // method to carryout the transpose
  //

  // Here we define a RAJA local array type.
  // The array type is templated on
  // 1) Data type
  // 2) Index permutation
  // 3) Dimensions of the array
  //

  using TILE_MEM =
    RAJA::LocalArray<int, RAJA::Perm<0, 1>, RAJA::SizeList<TILE_DIM, TILE_DIM>>;

  // **NOTE** Although the LocalArray is constructed
  // the array memory has not been allocated.

  TILE_MEM RAJA_Tile;

  //--------------------------------------------------------------------------//
  std::cout << "\n Running RAJA - with existing kernel API ...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  using SEQ_EXEC_POL =
    RAJA::KernelPolicy<
      RAJA::statement::Tile<1, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::loop_exec,
        RAJA::statement::Tile<0, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::loop_exec,

          //InitList identifies memory within the tuple which needs to be intialized
          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::InitList<2>,

          //
          //ForICount populates the Param<> arg within the local offset
          //
          RAJA::statement::ForICount<1, RAJA::statement::Param<0>, RAJA::loop_exec,
            RAJA::statement::ForICount<0, RAJA::statement::Param<1>, RAJA::loop_exec,
              RAJA::statement::Lambda<0>
            >
          >,

          RAJA::statement::ForICount<0, RAJA::statement::Param<1>, RAJA::loop_exec,
            RAJA::statement::ForICount<1, RAJA::statement::Param<0>, RAJA::loop_exec,
              RAJA::statement::Lambda<1>
            >
          >

          >
        >
      >
    >;

  //Existing interface requires lambdas to have all values from Segments + Parameters tuples
  RAJA::kernel_param<SEQ_EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment(0, N_c),
                                                     RAJA::RangeSegment(0, N_r)),
    //Param tuple holds arguments for the local index (offset) within the tiles + tile memory
    RAJA::make_tuple((int)0, (int)0, RAJA_Tile),

    //col - column of matrix, //row - row of matrix, //tx offset within tile, ty offset within tile
    [=](int col, int row, int tx, int ty, TILE_MEM &RAJA_Tile) {
      RAJA_Tile(ty, tx) = Aview(row, col);
    },

    [=](int col, int row, int tx, int ty, TILE_MEM &RAJA_Tile) {
      Atview(col, row) = RAJA_Tile(ty, tx);

  });

  checkResult<int>(Atview, N_c, N_r);

  //--------------------------------------------------------------------------//
  std::cout << "\n Running RAJA - with new kernel API ...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  using SEQ_EXEC_POL_NEW =
    RAJA::KernelPolicy<
      RAJA::statement::Tile<1, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::loop_exec,
        RAJA::statement::Tile<0, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::loop_exec,

          //InitList identifies memory within the tuple which needs to be intialized
          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::InitList<0>,

          RAJA::statement::For<1, RAJA::loop_exec,
            RAJA::statement::For<0, RAJA::loop_exec,
              //The additional types passed into the lambda statement
              //are used to specify lambda arguments.
              //Seg<0> - Uses the 0th entry in the Seg tuple as a lambda argument
              //Seg<1> - Uses the 1st entry in the Seg tuple as a lambda argument
              //Offset<0> - Uses the offset from the 0th seg tuple as a lambda argument
              //Offset<1> - Uses the offset from the 1th seg tuple as a lambda argument
              //Param<0> - Uses the 0th entry in the Param tuple as a lambda argument
              RAJA::statement::Lambda<0, Seg<0>, Seg<1>, OffSet<0>, OffSet<1>, Param<0> >
            >
          >,

          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              //If contiguous Lambda arguments are coming from the same tuple,
              //they may be merged into a *List<>
              RAJA::statement::Lambda<1, SegList<0, 1>, OffSetList<0, 1>, Param<0> >
            >
          >

          >
        >
      >
    >;

  RAJA::kernel_param<SEQ_EXEC_POL_NEW>( RAJA::make_tuple(RAJA::RangeSegment(0, N_c),
                                                        RAJA::RangeSegment(0, N_r)),

    RAJA::make_tuple(RAJA_Tile),

    [=](int col, int row, int tx, int ty, TILE_MEM &RAJA_Tile) {
        RAJA_Tile(ty, tx) = Aview(row, col);
    },

    [=](int col, int row, int tx, int ty, TILE_MEM &RAJA_Tile) {
      Atview(col, row) = RAJA_Tile(ty, tx);

  });

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);


  //--------------------------------------------------------------------------//
  std::cout << "\n Running RAJA - with existing kernel API 2 ...\n";
  //
  //The new interface enables having lambdas with different numbers of arguments
  //
  using NEW_POL =
    RAJA::KernelPolicy<
      RAJA::statement::For<0, RAJA::loop_exec,
        RAJA::statement::Lambda<0, Seg<0> >
      >,
      RAJA::statement::For<1, RAJA::loop_exec,
                           RAJA::statement::Lambda<1, Seg<1>, Param<0> >
      >
    >;

    RAJA::kernel_param<NEW_POL>(RAJA::make_tuple(RAJA::RangeSegment(5, 6),
                                           RAJA::RangeSegment(10, 11)),
                                RAJA::make_tuple((int) 5),
    [=] (int i) {
      printf("i = %d \n",i);
    },
    [=] (int j, int &val) {
      printf("j = %d, val = %d \n", j, val);
    });

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
      std::cout << "At(" << row << "," << col << ") = " << Atview(row, col)
                << std::endl;
    }
    std::cout << "" << std::endl;
  }
  std::cout << std::endl;
}
