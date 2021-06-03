//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include "halo-exchange.hpp"

#include "../memoryManager.hpp"


struct Extent
{
  int i_min;
  int i_max;
  int j_min;
  int j_max;
  int k_min;
  int k_max;
};

//
// Function to generate index lists for packing.
//
void create_pack_lists(std::vector<int*>& pack_index_lists,
                       std::vector<int >& pack_index_list_lengths,
                       const int halo_width, const int* grid_dims)
{
  std::vector<Extent> pack_index_list_extents(num_neighbors);

  // faces
  pack_index_list_extents[0]  = Extent{halo_width  , halo_width   + halo_width,
                                       halo_width  , grid_dims[1] + halo_width,
                                       halo_width  , grid_dims[2] + halo_width};
  pack_index_list_extents[1]  = Extent{grid_dims[0], grid_dims[0] + halo_width,
                                       halo_width  , grid_dims[1] + halo_width,
                                       halo_width  , grid_dims[2] + halo_width};
  pack_index_list_extents[2]  = Extent{halo_width  , grid_dims[0] + halo_width,
                                       halo_width  , halo_width   + halo_width,
                                       halo_width  , grid_dims[2] + halo_width};
  pack_index_list_extents[3]  = Extent{halo_width  , grid_dims[0] + halo_width,
                                       grid_dims[1], grid_dims[1] + halo_width,
                                       halo_width  , grid_dims[2] + halo_width};
  pack_index_list_extents[4]  = Extent{halo_width  , grid_dims[0] + halo_width,
                                       halo_width  , grid_dims[1] + halo_width,
                                       halo_width  , halo_width   + halo_width};
  pack_index_list_extents[5]  = Extent{halo_width  , grid_dims[0] + halo_width,
                                       halo_width  , grid_dims[1] + halo_width,
                                       grid_dims[2], grid_dims[2] + halo_width};

  // edges
  pack_index_list_extents[6]  = Extent{halo_width  , halo_width   + halo_width,
                                       halo_width  , halo_width   + halo_width,
                                       halo_width  , grid_dims[2] + halo_width};
  pack_index_list_extents[7]  = Extent{halo_width  , halo_width   + halo_width,
                                       grid_dims[1], grid_dims[1] + halo_width,
                                       halo_width  , grid_dims[2] + halo_width};
  pack_index_list_extents[8]  = Extent{grid_dims[0], grid_dims[0] + halo_width,
                                       halo_width  , halo_width   + halo_width,
                                       halo_width  , grid_dims[2] + halo_width};
  pack_index_list_extents[9]  = Extent{grid_dims[0], grid_dims[0] + halo_width,
                                       grid_dims[1], grid_dims[1] + halo_width,
                                       halo_width  , grid_dims[2] + halo_width};
  pack_index_list_extents[10] = Extent{halo_width  , halo_width   + halo_width,
                                       halo_width  , grid_dims[1] + halo_width,
                                       halo_width  , halo_width   + halo_width};
  pack_index_list_extents[11] = Extent{halo_width  , halo_width   + halo_width,
                                       halo_width  , grid_dims[1] + halo_width,
                                       grid_dims[2], grid_dims[2] + halo_width};
  pack_index_list_extents[12] = Extent{grid_dims[0], grid_dims[0] + halo_width,
                                       halo_width  , grid_dims[1] + halo_width,
                                       halo_width  , halo_width   + halo_width};
  pack_index_list_extents[13] = Extent{grid_dims[0], grid_dims[0] + halo_width,
                                       halo_width  , grid_dims[1] + halo_width,
                                       grid_dims[2], grid_dims[2] + halo_width};
  pack_index_list_extents[14] = Extent{halo_width  , grid_dims[0] + halo_width,
                                       halo_width  , halo_width   + halo_width,
                                       halo_width  , halo_width   + halo_width};
  pack_index_list_extents[15] = Extent{halo_width  , grid_dims[0] + halo_width,
                                       halo_width  , halo_width   + halo_width,
                                       grid_dims[2], grid_dims[2] + halo_width};
  pack_index_list_extents[16] = Extent{halo_width  , grid_dims[0] + halo_width,
                                       grid_dims[1], grid_dims[1] + halo_width,
                                       halo_width  , halo_width   + halo_width};
  pack_index_list_extents[17] = Extent{halo_width  , grid_dims[0] + halo_width,
                                       grid_dims[1], grid_dims[1] + halo_width,
                                       grid_dims[2], grid_dims[2] + halo_width};

  // corners
  pack_index_list_extents[18] = Extent{halo_width  , halo_width   + halo_width,
                                       halo_width  , halo_width   + halo_width,
                                       halo_width  , halo_width   + halo_width};
  pack_index_list_extents[19] = Extent{halo_width  , halo_width   + halo_width,
                                       halo_width  , halo_width   + halo_width,
                                       grid_dims[2], grid_dims[2] + halo_width};
  pack_index_list_extents[20] = Extent{halo_width  , halo_width   + halo_width,
                                       grid_dims[1], grid_dims[1] + halo_width,
                                       halo_width  , halo_width   + halo_width};
  pack_index_list_extents[21] = Extent{halo_width  , halo_width   + halo_width,
                                       grid_dims[1], grid_dims[1] + halo_width,
                                       grid_dims[2], grid_dims[2] + halo_width};
  pack_index_list_extents[22] = Extent{grid_dims[0], grid_dims[0] + halo_width,
                                       halo_width  , halo_width   + halo_width,
                                       halo_width  , halo_width   + halo_width};
  pack_index_list_extents[23] = Extent{grid_dims[0], grid_dims[0] + halo_width,
                                       halo_width  , halo_width   + halo_width,
                                       grid_dims[2], grid_dims[2] + halo_width};
  pack_index_list_extents[24] = Extent{grid_dims[0], grid_dims[0] + halo_width,
                                       grid_dims[1], grid_dims[1] + halo_width,
                                       halo_width  , halo_width   + halo_width};
  pack_index_list_extents[25] = Extent{grid_dims[0], grid_dims[0] + halo_width,
                                       grid_dims[1], grid_dims[1] + halo_width,
                                       grid_dims[2], grid_dims[2] + halo_width};

  const int grid_i_stride = 1;
  const int grid_j_stride = grid_dims[0] + 2*halo_width;
  const int grid_k_stride = grid_j_stride * (grid_dims[1] + 2*halo_width);

  for (int l = 0; l < num_neighbors; ++l) {

    Extent extent = pack_index_list_extents[l];

    pack_index_list_lengths[l] = (extent.i_max - extent.i_min) *
                                 (extent.j_max - extent.j_min) *
                                 (extent.k_max - extent.k_min) ;

    pack_index_lists[l] = memoryManager::allocate<int>(pack_index_list_lengths[l]);

    int* pack_list = pack_index_lists[l];

    int list_idx = 0;
    for (int kk = extent.k_min; kk < extent.k_max; ++kk) {
      for (int jj = extent.j_min; jj < extent.j_max; ++jj) {
        for (int ii = extent.i_min; ii < extent.i_max; ++ii) {

          int pack_idx = ii * grid_i_stride +
                         jj * grid_j_stride +
                         kk * grid_k_stride ;

          pack_list[list_idx] = pack_idx;

          list_idx += 1;
        }
      }
    }
  }
}

//
// Function to destroy packing index lists.
//
void destroy_pack_lists(std::vector<int*>& pack_index_lists)
{
  for (int l = 0; l < num_neighbors; ++l) {
    memoryManager::deallocate(pack_index_lists[l]);
  }
}


//
// Function to generate index lists for unpacking.
//
void create_unpack_lists(std::vector<int*>& unpack_index_lists, std::vector<int>& unpack_index_list_lengths,
                         const int halo_width, const int* grid_dims)
{
  std::vector<Extent> unpack_index_list_extents(num_neighbors);

  // faces
  unpack_index_list_extents[0]  = Extent{0                        ,                  halo_width,
                                         halo_width               , grid_dims[1] +   halo_width,
                                         halo_width               , grid_dims[2] +   halo_width};
  unpack_index_list_extents[1]  = Extent{grid_dims[0] + halo_width, grid_dims[0] + 2*halo_width,
                                         halo_width               , grid_dims[1] +   halo_width,
                                         halo_width               , grid_dims[2] +   halo_width};
  unpack_index_list_extents[2]  = Extent{halo_width               , grid_dims[0] +   halo_width,
                                         0                        ,                  halo_width,
                                         halo_width               , grid_dims[2] +   halo_width};
  unpack_index_list_extents[3]  = Extent{halo_width               , grid_dims[0] +   halo_width,
                                         grid_dims[1] + halo_width, grid_dims[1] + 2*halo_width,
                                         halo_width               , grid_dims[2] +   halo_width};
  unpack_index_list_extents[4]  = Extent{halo_width               , grid_dims[0] +   halo_width,
                                         halo_width               , grid_dims[1] +   halo_width,
                                         0                        ,                  halo_width};
  unpack_index_list_extents[5]  = Extent{halo_width               , grid_dims[0] +   halo_width,
                                         halo_width               , grid_dims[1] +   halo_width,
                                         grid_dims[2] + halo_width, grid_dims[2] + 2*halo_width};

  // edges
  unpack_index_list_extents[6]  = Extent{0                        ,                  halo_width,
                                         0                        ,                  halo_width,
                                         halo_width               , grid_dims[2] +   halo_width};
  unpack_index_list_extents[7]  = Extent{0                        ,                  halo_width,
                                         grid_dims[1] + halo_width, grid_dims[1] + 2*halo_width,
                                         halo_width               , grid_dims[2] +   halo_width};
  unpack_index_list_extents[8]  = Extent{grid_dims[0] + halo_width, grid_dims[0] + 2*halo_width,
                                         0                        ,                  halo_width,
                                         halo_width               , grid_dims[2] +   halo_width};
  unpack_index_list_extents[9]  = Extent{grid_dims[0] + halo_width, grid_dims[0] + 2*halo_width,
                                         grid_dims[1] + halo_width, grid_dims[1] + 2*halo_width,
                                         halo_width               , grid_dims[2] +   halo_width};
  unpack_index_list_extents[10] = Extent{0                        ,                  halo_width,
                                         halo_width               , grid_dims[1] +   halo_width,
                                         0                        ,                  halo_width};
  unpack_index_list_extents[11] = Extent{0                        ,                  halo_width,
                                         halo_width               , grid_dims[1] +   halo_width,
                                         grid_dims[2] + halo_width, grid_dims[2] + 2*halo_width};
  unpack_index_list_extents[12] = Extent{grid_dims[0] + halo_width, grid_dims[0] + 2*halo_width,
                                         halo_width               , grid_dims[1] +   halo_width,
                                         0                        ,                  halo_width};
  unpack_index_list_extents[13] = Extent{grid_dims[0] + halo_width, grid_dims[0] + 2*halo_width,
                                         halo_width               , grid_dims[1] +   halo_width,
                                         grid_dims[2] + halo_width, grid_dims[2] + 2*halo_width};
  unpack_index_list_extents[14] = Extent{halo_width               , grid_dims[0] +   halo_width,
                                         0                        ,                  halo_width,
                                         0                        ,                  halo_width};
  unpack_index_list_extents[15] = Extent{halo_width               , grid_dims[0] +   halo_width,
                                         0                        ,                  halo_width,
                                         grid_dims[2] + halo_width, grid_dims[2] + 2*halo_width};
  unpack_index_list_extents[16] = Extent{halo_width               , grid_dims[0] +   halo_width,
                                         grid_dims[1] + halo_width, grid_dims[1] + 2*halo_width,
                                         0                        ,                  halo_width};
  unpack_index_list_extents[17] = Extent{halo_width               , grid_dims[0] +   halo_width,
                                         grid_dims[1] + halo_width, grid_dims[1] + 2*halo_width,
                                         grid_dims[2] + halo_width, grid_dims[2] + 2*halo_width};

  // corners
  unpack_index_list_extents[18] = Extent{0                        ,                  halo_width,
                                         0                        ,                  halo_width,
                                         0                        ,                  halo_width};
  unpack_index_list_extents[19] = Extent{0                        ,                  halo_width,
                                         0                        ,                  halo_width,
                                         grid_dims[2] + halo_width, grid_dims[2] + 2*halo_width};
  unpack_index_list_extents[20] = Extent{0                        ,                  halo_width,
                                         grid_dims[1] + halo_width, grid_dims[1] + 2*halo_width,
                                         0                        ,                  halo_width};
  unpack_index_list_extents[21] = Extent{0                        ,                  halo_width,
                                         grid_dims[1] + halo_width, grid_dims[1] + 2*halo_width,
                                         grid_dims[2] + halo_width, grid_dims[2] + 2*halo_width};
  unpack_index_list_extents[22] = Extent{grid_dims[0] + halo_width, grid_dims[0] + 2*halo_width,
                                         0                        ,                  halo_width,
                                         0                        ,                  halo_width};
  unpack_index_list_extents[23] = Extent{grid_dims[0] + halo_width, grid_dims[0] + 2*halo_width,
                                         0                        ,                  halo_width,
                                         grid_dims[2] + halo_width, grid_dims[2] + 2*halo_width};
  unpack_index_list_extents[24] = Extent{grid_dims[0] + halo_width, grid_dims[0] + 2*halo_width,
                                         grid_dims[1] + halo_width, grid_dims[1] + 2*halo_width,
                                         0                        ,                  halo_width};
  unpack_index_list_extents[25] = Extent{grid_dims[0] + halo_width, grid_dims[0] + 2*halo_width,
                                         grid_dims[1] + halo_width, grid_dims[1] + 2*halo_width,
                                         grid_dims[2] + halo_width, grid_dims[2] + 2*halo_width};

  const int grid_i_stride = 1;
  const int grid_j_stride = grid_dims[0] + 2*halo_width;
  const int grid_k_stride = grid_j_stride * (grid_dims[1] + 2*halo_width);

  for (int l = 0; l < num_neighbors; ++l) {

    Extent extent = unpack_index_list_extents[l];

    unpack_index_list_lengths[l] = (extent.i_max - extent.i_min) *
                                   (extent.j_max - extent.j_min) *
                                   (extent.k_max - extent.k_min) ;

    unpack_index_lists[l] = memoryManager::allocate<int>(unpack_index_list_lengths[l]);

    int* unpack_list = unpack_index_lists[l];

    int list_idx = 0;
    for (int kk = extent.k_min; kk < extent.k_max; ++kk) {
      for (int jj = extent.j_min; jj < extent.j_max; ++jj) {
        for (int ii = extent.i_min; ii < extent.i_max; ++ii) {

          int unpack_idx = ii * grid_i_stride +
                           jj * grid_j_stride +
                           kk * grid_k_stride ;

          unpack_list[list_idx] = unpack_idx;

          list_idx += 1;
        }
      }
    }
  }
}

//
// Function to destroy unpacking index lists.
//
void destroy_unpack_lists(std::vector<int*>& unpack_index_lists)
{
  for (int l = 0; l < num_neighbors; ++l) {
    memoryManager::deallocate(unpack_index_lists[l]);
  }
}
