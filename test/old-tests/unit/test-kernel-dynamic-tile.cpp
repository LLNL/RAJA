//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"

TEST(KernelDynamicTile, Tile2D) {
    
  const int DIM_X = 7;
  const int DIM_Y = 11;
  const int TILE_X = 3;
  const int TILE_Y = 5;

  double* expected = new double[DIM_X * DIM_Y];
  RAJA::View<double, RAJA::Layout<2> > expectedView(expected, DIM_Y, DIM_X);
  double* actual = new double[DIM_X * DIM_Y];
  RAJA::View<double, RAJA::Layout<2> > actualView(actual, DIM_Y, DIM_X);

  using ExecPolicy = RAJA::KernelPolicy< 
      RAJA::statement::Tile<1, RAJA::tile_dynamic<1>, RAJA::seq_exec,
        RAJA::statement::Tile<0, RAJA::tile_dynamic<0>, RAJA::seq_exec,
          RAJA::statement::For<1, RAJA::seq_exec, 
            RAJA::statement::For<0, RAJA::seq_exec,
              RAJA::statement::Lambda<0, RAJA::Segs<0, 1>, RAJA::Params<>>>>>>>;

  int i = 0;
  RAJA::kernel_param<ExecPolicy>(
    RAJA::make_tuple(RAJA::RangeSegment{0,DIM_X}, RAJA::RangeSegment{0,DIM_Y}),
    RAJA::make_tuple(RAJA::TileSize{TILE_X}, RAJA::TileSize{TILE_Y}),
    [actualView, &i] (int x, int y) {
      actualView(y, x) = i++;
    });

  i = 0;
  for (int tile_y = 0; tile_y < DIM_Y; tile_y += TILE_Y) {
    for (int tile_x = 0; tile_x < DIM_X; tile_x += TILE_X) {
      for (int y = tile_y; y < std::min({tile_y + TILE_Y, DIM_Y}); ++y) {
        for (int x = tile_x; x < std::min({tile_x + TILE_X, DIM_X}); ++x) {
          expectedView(y, x) = i++;
        }
      }
    }
  }

  for (int idx = 0; idx < DIM_X * DIM_Y; ++idx) {
    ASSERT_EQ(actual[idx], expected[idx]) << "Vectors x and y differ at index " << idx;
  }

  delete[] expected;
  delete[] actual;
}

TEST (KernelDynamicTile, Tile3D) { 

  const int DIM_X = 7;
  const int DIM_Y = 11;
  const int DIM_Z = 13;
  const int TILE_X = 3;
  const int TILE_Y = 5;
  const int TILE_Z = 7;

  double* expected = new double[DIM_X * DIM_Y * DIM_Z];
  RAJA::View<double, RAJA::Layout<3> > expectedView(expected, DIM_Z, DIM_Y, DIM_X);
  double* actual = new double[DIM_X * DIM_Y * DIM_Z];
  RAJA::View<double, RAJA::Layout<3> > actualView(actual, DIM_Z, DIM_Y, DIM_X);

  using ExecPolicy = RAJA::KernelPolicy< 
      RAJA::statement::Tile<2, RAJA::tile_dynamic<2>, RAJA::seq_exec,
        RAJA::statement::Tile<1, RAJA::tile_dynamic<1>, RAJA::seq_exec,
          RAJA::statement::Tile<0, RAJA::tile_dynamic<0>, RAJA::seq_exec,
            RAJA::statement::For<2, RAJA::seq_exec, 
              RAJA::statement::For<1, RAJA::seq_exec, 
                RAJA::statement::For<0, RAJA::seq_exec,
                  RAJA::statement::Lambda<0, RAJA::Segs<0, 1, 2>, RAJA::Params<>>>>>>>>>;

  int i = 0;
  RAJA::kernel_param<ExecPolicy>(
    RAJA::make_tuple(RAJA::RangeSegment{0,DIM_X}, RAJA::RangeSegment{0,DIM_Y}, RAJA::RangeSegment{0,DIM_Z}),
    RAJA::make_tuple(RAJA::TileSize{TILE_X}, RAJA::TileSize{TILE_Y}, RAJA::TileSize{TILE_Z}),
    [actualView, &i] (int x, int y, int z) {
      actualView(z, y, x) = ++i;
    });

  i = 0;
  for (int tile_z = 0; tile_z < DIM_Z; tile_z += TILE_Z) {
    for (int tile_y = 0; tile_y < DIM_Y; tile_y += TILE_Y) {
      for (int tile_x = 0; tile_x < DIM_X; tile_x += TILE_X) {
        for (int z = tile_z; z < std::min({tile_z + TILE_Z, DIM_Z}); ++z) {
          for (int y = tile_y; y < std::min({tile_y + TILE_Y, DIM_Y}); ++y) {
            for (int x = tile_x; x < std::min({tile_x + TILE_X, DIM_X}); ++x) {
              expectedView(z, y, x) = ++i;
            }
          }
        }
      }
    }
  }

  for (int idx = 0; idx < DIM_X * DIM_Y * DIM_Z; ++idx) {
    ASSERT_EQ(actual[idx], expected[idx]) << "Vectors x and y differ at index " << idx;
  }

  delete[] expected;
  delete[] actual;
}