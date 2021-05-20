//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <limits>
#include <vector>

#include "halo-exchange.hpp"
#include "Schedule.hpp"
#include "CopyTransaction.hpp"
#include "../memoryManager.hpp"
#include "loop.hpp"

#include "RAJA/util/Timer.hpp"


/*
 *  Halo exchange Example
 *
 *  Packs and Unpacks data from 3D variables as is done in a halo exchange.
 *  It illustrates how to use the workgroup set of constructs.
 *
 *  RAJA features shown:
 *    - `WorkPool` template object
 *    - `WorkGroup` template object
 *    - `WorkSite` template object
 *    -  Index range segment
 *    -  WorkGroup policies
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */


//
// Functions for checking and printing results
//
void checkResult(std::vector<double*> const& vars, std::vector<double*> const& vars_ref,
                 int var_size, int num_vars);
void printResult(std::vector<double*> const& vars, int var_size, int num_vars);


int main(int argc, char **argv)
{

  std::cout << "\n\nRAJA halo exchange example...\n";

  if (argc != 1 && argc != 7) {
    std::cerr << "Usage: tut_halo-exchange "
              << "[grid_x grid_y grid_z halo_width num_vars num_cycles]\n";
    std::exit(1);
  }

  //
  // Define grid dimensions
  // Define halo width
  // Define number of grid variables
  // Define number of cycles
  //
  const int grid_dims[3] = { (argc != 7) ? 100 : std::atoi(argv[1]),
                             (argc != 7) ? 100 : std::atoi(argv[2]),
                             (argc != 7) ? 100 : std::atoi(argv[3]) };
  const int halo_width =     (argc != 7) ?   1 : std::atoi(argv[4]);
  const int num_vars   =     (argc != 7) ?   3 : std::atoi(argv[5]);
  const int num_cycles =     (argc != 7) ? 128 : std::atoi(argv[6]);

  std::cout << "grid dimensions "     << grid_dims[0]
            << " x "                  << grid_dims[1]
            << " x "                  << grid_dims[2] << "\n"
            << "halo width "          << halo_width   << "\n"
            << "number of variables " << num_vars     << "\n"
            << "number of cycles "    << num_cycles   << "\n";

  if ( grid_dims[0] < halo_width ||
       grid_dims[1] < halo_width ||
       grid_dims[2] < halo_width ) {
    std::cerr << "Error: "
              << "grid dimensions must not be smaller than the halo width\n";
    std::exit(1);
  }

  const int grid_plus_halo_dims[3] = { grid_dims[0] + 2*halo_width,
                                       grid_dims[1] + 2*halo_width,
                                       grid_dims[2] + 2*halo_width };

  const int var_size = grid_plus_halo_dims[0] *
                       grid_plus_halo_dims[1] *
                       grid_plus_halo_dims[2] ;

  //
  // Allocate grid variables and reference grid variables used to check
  // correctness.
  //
  std::vector<double*> vars    (num_vars, nullptr);
  std::vector<double*> vars_ref(num_vars, nullptr);

  for (int v = 0; v < num_vars; ++v) {
    vars[v]     = memoryManager::allocate<double>(var_size);
    vars_ref[v] = memoryManager::allocate<double>(var_size);
  }


  //
  // Generate index lists for packing and unpacking
  //
  std::vector<int*> pack_index_lists(num_neighbors, nullptr);
  std::vector<int > pack_index_list_lengths(num_neighbors, 0);
  create_pack_lists(pack_index_lists, pack_index_list_lengths, halo_width, grid_dims);

  std::vector<int*> unpack_index_lists(num_neighbors, nullptr);
  std::vector<int > unpack_index_list_lengths(num_neighbors, 0);
  create_unpack_lists(unpack_index_lists, unpack_index_list_lengths, halo_width, grid_dims);


  //
  // Convenience type alias to reduce typing
  //
  using range_segment = RAJA::TypedRangeSegment<int>;


  TimerStats timer;


//----------------------------------------------------------------------------//
  for (int p = static_cast<int>(LoopPattern::seq);
       p < static_cast<int>(LoopPattern::End); ++p) {

    LoopPattern pattern = static_cast<LoopPattern>(p);

    SetLoopPatternScope sepc(pattern);

    std::cout << "\n Running " << get_loop_pattern_name() << " halo exchange...\n";


    // allocate per pattern memory
    RAJA::resources::Resource res = get_loop_pattern_resource();

    std::vector<double*> pattern_vars(num_vars, nullptr);
    std::vector<int*>    pattern_pack_index_lists(num_neighbors, nullptr);
    std::vector<int*>    pattern_unpack_index_lists(num_neighbors, nullptr);

    for (int v = 0; v < num_vars; ++v) {
      pattern_vars[v] = res.allocate<double>(var_size);
    }

    for (int l = 0; l < num_neighbors; ++l) {
      int pack_len = pack_index_list_lengths[l];
      pattern_pack_index_lists[l] = res.allocate<int>(pack_len);
      res.memcpy(pattern_pack_index_lists[l], pack_index_lists[l], pack_len * sizeof(int));

      int unpack_len = unpack_index_list_lengths[l];
      pattern_unpack_index_lists[l] = res.allocate<int>(unpack_len);
      res.memcpy(pattern_unpack_index_lists[l], unpack_index_lists[l], unpack_len * sizeof(int));
    }


    const int my_rank = 0;
    Schedule schedule(my_rank);

    // populate schedule
    for (int l = 0; l < num_neighbors; ++l) {

      int neighbor_rank = l + 1;

      for (double* var : pattern_vars) {

        int* pack_list = pattern_pack_index_lists[l];
        int  pack_len  = pack_index_list_lengths[l];

        int* recv_list = pattern_unpack_index_lists[l];
        int  recv_len  = unpack_index_list_lengths[l];

        CopyTransaction* recv =
            new CopyTransaction(neighbor_rank,
                                my_rank,
                                var,
                                recv_list, recv_len);

        CopyTransaction* pack =
            new CopyTransaction(my_rank,
                                neighbor_rank,
                                var,
                                pack_list, pack_len);

        if (get_loop_pattern_fusible()) {
          schedule.appendTransaction(std::unique_ptr<FusibleTransaction>(recv));
          schedule.appendTransaction(std::unique_ptr<FusibleTransaction>(pack));
        } else {
          schedule.appendTransaction(std::unique_ptr<Transaction>(recv));
          schedule.appendTransaction(std::unique_ptr<Transaction>(pack));
        }

      }

    }

    for (int c = 0; c < num_cycles; ++c ) {
      timer.start();
      {

        // set vars
        for (int v = 0; v < num_vars; ++v) {

          double* var = pattern_vars[v];

          loop(var_size, [=] RAJA_HOST_DEVICE (int i) {
            var[i] = i + v;
          });
        }

        schedule.communicate();

      }
      timer.stop();
    }

    // deallocate per pattern memory
    for (int v = 0; v < num_vars; ++v) {
      res.memcpy(vars[v], pattern_vars[v], var_size * sizeof(double));
      res.deallocate(pattern_vars[v]);
    }

    for (int l = 0; l < num_neighbors; ++l) {
      res.deallocate(pattern_pack_index_lists[l]);
      res.deallocate(pattern_unpack_index_lists[l]);
    }


    std::cout<< "\t" << timer.get_num() << " cycles" << std::endl;
    std::cout<< "\tavg cycle run time " << timer.get_avg() << " seconds" << std::endl;
    std::cout<< "\tmin cycle run time " << timer.get_min() << " seconds" << std::endl;
    std::cout<< "\tmax cycle run time " << timer.get_max() << " seconds" << std::endl;
    timer.reset();

    if (pattern == LoopPattern::seq) {
      // copy result of exchange for reference later
      for (int v = 0; v < num_vars; ++v) {
        res.memcpy(vars_ref[v], vars[v], var_size * sizeof(double));
      }
    } else {
      // check results against reference copy
      checkResult(vars, vars_ref, var_size, num_vars);
      //printResult(vars, var_size, num_vars);
    }
  }


//
// Clean up.
//
  for (int v = 0; v < num_vars; ++v) {
    memoryManager::deallocate(vars[v]);
    memoryManager::deallocate(vars_ref[v]);
  }

  destroy_pack_lists(pack_index_lists);
  destroy_unpack_lists(unpack_index_lists);


  std::cout << "\n DONE!...\n";

  return 0;
}


//
// Function to compare result to reference and report P/F.
//
void checkResult(std::vector<double*> const& vars, std::vector<double*> const& vars_ref,
                 int var_size, int num_vars)
{
  bool correct = true;
  for (int v = 0; v < num_vars; ++v) {
    double* var = vars[v];
    double* var_ref = vars_ref[v];
    for (int i = 0; i < var_size; i++) {
      if ( var[i] != var_ref[i] ) { correct = false; }
    }
  }
  if ( correct ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
}

//
// Function to print result.
//
void printResult(std::vector<double*> const& vars, int var_size, int num_vars)
{
  std::cout << std::endl;
  for (int v = 0; v < num_vars; ++v) {
    double* var = vars[v];
    for (int i = 0; i < var_size; i++) {
      std::cout << "result[" << i << "] = " << var[i] << std::endl;
    }
  }
  std::cout << std::endl;
}

