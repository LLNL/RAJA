//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_EXAMPLES_HALOEXCHANGE_HALOEXCHANGE_HPP
#define RAJA_EXAMPLES_HALOEXCHANGE_HALOEXCHANGE_HPP

#include <cstdlib>
#include <vector>

#include "RAJA/RAJA.hpp"

/*
  num_neighbors - specifies the number of neighbors that each process would be
                  communicating with in 3D halo exchange
*/
const int num_neighbors = 26;

//
// Functions for allocating and populating packing and unpacking lists
//
extern void create_pack_lists(std::vector<int*>& pack_index_lists,
                              std::vector<int>& pack_index_list_lengths,
                              const int halo_width,
                              const int* grid_dims);
extern void create_unpack_lists(std::vector<int*>& unpack_index_lists,
                                std::vector<int>& unpack_index_list_lengths,
                                const int halo_width,
                                const int* grid_dims);
extern void destroy_pack_lists(std::vector<int*>& pack_index_lists);
extern void destroy_unpack_lists(std::vector<int*>& unpack_index_lists);

#endif // RAJA_EXAMPLES_HALOEXCHANGE_HALOEXCHANGE_HPP
