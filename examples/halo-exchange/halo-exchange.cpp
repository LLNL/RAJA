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
#include <cstring>
#include <cmath>

#include "../memoryManager.hpp"
#include "halo-exchange.hpp"
#include "Point.hpp"
#include "Item.hpp"
#include "Schedule.hpp"
#include "loop.hpp"

#include "RAJA/util/Timer.hpp"


/*
 *  Halo exchange Example
 *
 *  Packs and Unpacks data from 3D variables as is done in a halo exchange.
 */


//
// Functions for checking and printing results
//
void checkResult(std::vector<double*> const& vars, std::vector<double*> const& vars_ref,
                 int var_size, int num_vars);
void printResult(std::vector<double*> const& vars, int var_size, int num_vars);


int main(int argc, char **argv)
{

  std::cout << "RAJA halo exchange example..." << std::endl;

  if (argc != 1 && argc != 8) {
    std::cerr << "Usage: tut_halo-exchange "
              << "[grid_x grid_y grid_z halo_width num_vars num_cycles transaction_type(copy or sum)]" << std::endl;
    std::exit(1);
  }

  //
  // Define grid dimensions
  // Define halo width
  // Define number of grid variables
  // Define number of cycles
  //
  const int grid_dims[3] = { (argc != 8) ? 100 : std::atoi(argv[1]),
                             (argc != 8) ? 100 : std::atoi(argv[2]),
                             (argc != 8) ? 100 : std::atoi(argv[3]) };
  const int halo_width =     (argc != 8) ?   1 : std::atoi(argv[4]);
  const int num_vars   =     (argc != 8) ?   3 : std::atoi(argv[5]);
  const int num_cycles =     (argc != 8) ? 128 : std::atoi(argv[6]);
  const TransactionType transaction_type = (argc != 8) ? TransactionType::copy
                                            : (std::strcmp(argv[7], "copy") == 0) ? TransactionType::copy
                                            : (std::strcmp(argv[7], "sum") == 0) ? TransactionType::sum
                                            : TransactionType::invalid;

  std::cout << "grid dimensions "     << grid_dims[0]
                             << " x " << grid_dims[1]
                             << " x " << grid_dims[2] << "\n"
            << "halo width "          << halo_width   << "\n"
            << "number of variables " << num_vars     << "\n"
            << "number of cycles "    << num_cycles   << "\n"
            << "transaction type "    << get_transaction_type_name(transaction_type) << "\n";

  if ( grid_dims[0] < halo_width ||
       grid_dims[1] < halo_width ||
       grid_dims[2] < halo_width ) {
    std::cerr << "Error: "
              << "grid dimensions must not be smaller than the halo width" << std::endl;
    std::exit(1);
  }

  if ( transaction_type == TransactionType::invalid) {
    std::cerr << "Error: "
              << "transaction type must be valid" << std::endl;
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

  if (transaction_type == TransactionType::sum)
  {
    // simulate aggregating contributions from neighbors by swapping
    // pack and unpack lists for sum transactions
    unpack_index_lists.swap(pack_index_lists);
    unpack_index_list_lengths.swap(pack_index_list_lengths);
  }

  std::cout << std::endl;


  TimerStats timer;


//----------------------------------------------------------------------------//
  {
    std::cout << "Running Simple C-style halo exchange " << get_transaction_type_name(transaction_type) << " ...\n"
              << "  ordering: pack:   items " << get_order_name(Order::ordered) << ", transactions " << get_order_name(Order::ordered) << "\n"
              << "            unpack: items " << get_order_name(Order::ordered) << ", transactions " << get_order_name(Order::ordered) << std::endl;


    std::vector<double*> buffers(num_neighbors, nullptr);

    for (int l = 0; l < num_neighbors; ++l) {

      int buffer_len = num_vars * pack_index_list_lengths[l];

      buffers[l] = memoryManager::allocate<double>(buffer_len);

    }

    if (transaction_type == TransactionType::copy) {
      for (int c = 0; c < num_cycles; ++c ) {
        timer.start();
        {
          // set vars
          for (int v = 0; v < num_vars; ++v) {

            double* var = vars[v];

            for (int i = 0; i < var_size; i++) {
              var[i] = sqrt(i + v);
            }
          }

          for (int l = 0; l < num_neighbors; ++l) {

            double* buffer = buffers[l];
            int* list = pack_index_lists[l];
            int  len  = pack_index_list_lengths[l];

            // pack
            for (int v = 0; v < num_vars; ++v) {

              double* var = vars[v];

              for (int i = 0; i < len; i++) {
                buffer[i] = var[list[i]];
              }

              buffer += len;
            }

            // send single message
          }

          for (int l = 0; l < num_neighbors; ++l) {

            // recv single message

            double* buffer = buffers[l];
            int* list = unpack_index_lists[l];
            int  len  = unpack_index_list_lengths[l];

            // unpack
            for (int v = 0; v < num_vars; ++v) {

              double* var = vars[v];

              for (int i = 0; i < len; i++) {
                var[list[i]] = buffer[i];
              }

              buffer += len;
            }
          }

        }
        timer.stop();
      }
    }
    else if (transaction_type == TransactionType::sum) {
      for (int c = 0; c < num_cycles; ++c ) {
        timer.start();
        {
          // set vars
          for (int v = 0; v < num_vars; ++v) {

            double* var = vars[v];

            for (int i = 0; i < var_size; i++) {
              var[i] = sqrt(i + v);
            }
          }

          for (int l = 0; l < num_neighbors; ++l) {

            double* buffer = buffers[l];
            int* list = pack_index_lists[l];
            int  len  = pack_index_list_lengths[l];

            // pack
            for (int v = 0; v < num_vars; ++v) {

              double* var = vars[v];

              for (int i = 0; i < len; i++) {
                buffer[i] = var[list[i]];
              }

              buffer += len;
            }

            // send single message
          }

          for (int l = 0; l < num_neighbors; ++l) {

            // recv single message

            double* buffer = buffers[l];
            int* list = unpack_index_lists[l];
            int  len  = unpack_index_list_lengths[l];

            // unpack
            for (int v = 0; v < num_vars; ++v) {

              double* var = vars[v];

              for (int i = 0; i < len; i++) {
                var[list[i]] += buffer[i];
              }

              buffer += len;
            }
          }

        }
        timer.stop();
      }
    }
    else {
      assert(0); // error
    }

    for (int l = 0; l < num_neighbors; ++l) {

      memoryManager::deallocate(buffers[l]);

    }

    std::cout << "    " << timer.get_num() << " cycles\n";
    std::cout << "    avg cycle run time " << timer.get_avg() << " seconds\n";
    std::cout << "    min cycle run time " << timer.get_min() << " seconds\n";
    std::cout << "    max cycle run time " << timer.get_max() << " seconds\n";
    timer.reset();

    // copy result of exchange for reference later
    for (int v = 0; v < num_vars; ++v) {

      double* var     = vars[v];
      double* var_ref = vars_ref[v];

      for (int i = 0; i < var_size; i++) {
        var_ref[i] = var[i];
      }
    }
    std::cout << std::endl;
  }


//----------------------------------------------------------------------------//
  for (int p = static_cast<int>(LoopPattern::seq);
       p < static_cast<int>(LoopPattern::End); ++p) {

    LoopPattern pattern = static_cast<LoopPattern>(p);

    SetLoopPatternScope sepc(pattern);

    // allow unordered unpacking with copy, but not sum
    const int num_orderings = (transaction_type == TransactionType::copy) ? 2
                            : (transaction_type == TransactionType::sum)  ? 1
                            : 0;
    for (int ordering = 0; ordering < num_orderings; ++ordering) {

      const Order order_pack_transactions   =                                      Order::unordered;
      const Order order_unpack_transactions = ordering == 0 ? Order::ordered     : Order::unordered;
      const Order order_pack_items          =                                      Order::unordered;
      const Order order_unpack_items        = ordering == 0 ? Order::ordered     : Order::unordered;


      std::cout << "Running Schedule " << get_loop_pattern_name() << " halo exchange " << get_transaction_type_name(transaction_type) << " ...\n"
                << "  ordering: pack:   items " << get_order_name(order_pack_items) << ", transactions " << get_order_name(order_pack_transactions) << "\n"
                << "            unpack: items " << get_order_name(order_unpack_items) << ", transactions " << get_order_name(order_unpack_transactions) << std::endl;


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

      Point point;

      // populate point
      {
        std::vector<typename Point::item_id_type> item_ids;

        for (size_t v = 0; v < pattern_vars.size(); ++v) {

          item_ids.emplace_back(
              point.addItem(pattern_vars[v],
                            order_pack_transactions,
                            pattern_pack_index_lists,
                            pack_index_list_lengths,
                            order_unpack_transactions,
                            pattern_unpack_index_lists,
                            unpack_index_list_lengths,
                            transaction_type));

          if (order_pack_items != Order::unordered && v > 0u) {
            point.addPackDependency(item_ids[v-1], item_ids[v]);
          }
          if (order_unpack_items != Order::unordered && v > 0u) {
            point.addUnpackDependency(item_ids[v-1], item_ids[v]);
          }

        }

        point.createSchedule();
      }

      for (int c = 0; c < num_cycles; ++c ) {
        timer.start();
        {

          // set vars
          for (int v = 0; v < num_vars; ++v) {

            double* var = pattern_vars[v];

            loop(var_size, [=] RAJA_HOST_DEVICE (int i) {
              var[i] = sqrt(i + v);
            });
          }

          point.getSchedule().communicate();

        }
        timer.stop();
      }

      point.clear();

      // deallocate per pattern memory
      for (int v = 0; v < num_vars; ++v) {
        res.memcpy(vars[v], pattern_vars[v], var_size * sizeof(double));
        res.deallocate(pattern_vars[v]);
      }

      for (int l = 0; l < num_neighbors; ++l) {
        res.deallocate(pattern_pack_index_lists[l]);
        res.deallocate(pattern_unpack_index_lists[l]);
      }


      std::cout << "    " << timer.get_num() << " cycles\n";
      std::cout << "    avg cycle run time " << timer.get_avg() << " seconds\n";
      std::cout << "    min cycle run time " << timer.get_min() << " seconds\n";
      std::cout << "    max cycle run time " << timer.get_max() << " seconds\n";
      timer.reset();

      // check results against reference copy
      checkResult(vars, vars_ref, var_size, num_vars);
      //printResult(vars, var_size, num_vars);
      std::cout << std::endl;
    }
  }


//----------------------------------------------------------------------------//
  for (int p = static_cast<int>(LoopPattern::seq);
       p < static_cast<int>(LoopPattern::End); ++p) {

    LoopPattern pattern = static_cast<LoopPattern>(p);

    SetLoopPatternScope sepc(pattern);

    // allow unordered unpacking with copy, but not sum
    const int num_orderings = (transaction_type == TransactionType::copy) ? 2
                            : (transaction_type == TransactionType::sum)  ? 1
                            : 0;
    for (int ordering = 0; ordering < num_orderings; ++ordering) {

      const Order order_pack_transactions   =                                      Order::unordered;
      const Order order_unpack_transactions = ordering == 0 ? Order::ordered     : Order::unordered;
      const Order order_pack_items          =                                      Order::unordered;
      const Order order_unpack_items        = ordering == 0 ? Order::ordered     : Order::unordered;


      std::cout << "Running GraphSchedule " << get_loop_pattern_name() << " halo exchange " << get_transaction_type_name(transaction_type) << " ...\n"
                << "  ordering: pack:   items " << get_order_name(order_pack_items) << ", transactions " << get_order_name(order_pack_transactions) << "\n"
                << "            unpack: items " << get_order_name(order_unpack_items) << ", transactions " << get_order_name(order_unpack_transactions) << std::endl;


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

      Point point;

      // populate point
      {
        std::vector<typename Point::item_id_type> item_ids;

        for (size_t v = 0; v < pattern_vars.size(); ++v) {

          item_ids.emplace_back(
              point.addItem(pattern_vars[v],
                            order_pack_transactions,
                            pattern_pack_index_lists,
                            pack_index_list_lengths,
                            order_unpack_transactions,
                            pattern_unpack_index_lists,
                            unpack_index_list_lengths,
                            transaction_type));

          if (order_pack_items != Order::unordered && v > 0u) {
            point.addPackDependency(item_ids[v-1], item_ids[v]);
          }
          if (order_unpack_items != Order::unordered && v > 0u) {
            point.addUnpackDependency(item_ids[v-1], item_ids[v]);
          }

        }

        point.createGraphSchedule();
      }

      for (int c = 0; c < num_cycles; ++c ) {
        timer.start();
        {

          // set vars
          for (int v = 0; v < num_vars; ++v) {

            double* var = pattern_vars[v];

            loop(var_size, [=] RAJA_HOST_DEVICE (int i) {
              var[i] = sqrt(i + v);
            });
          }

          point.getGraphSchedule().communicate();

        }
        timer.stop();
      }

      point.clear();

      // deallocate per pattern memory
      for (int v = 0; v < num_vars; ++v) {
        res.memcpy(vars[v], pattern_vars[v], var_size * sizeof(double));
        res.deallocate(pattern_vars[v]);
      }

      for (int l = 0; l < num_neighbors; ++l) {
        res.deallocate(pattern_pack_index_lists[l]);
        res.deallocate(pattern_unpack_index_lists[l]);
      }


      std::cout << "    " << timer.get_num() << " cycles\n";
      std::cout << "    avg cycle run time " << timer.get_avg() << " seconds\n";
      std::cout << "    min cycle run time " << timer.get_min() << " seconds\n";
      std::cout << "    max cycle run time " << timer.get_max() << " seconds\n";
      timer.reset();

      // check results against reference copy
      checkResult(vars, vars_ref, var_size, num_vars);
      //printResult(vars, var_size, num_vars);
      std::cout << std::endl;
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


  std::cout << "DONE!..." << std::endl;

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
    std::cout << "  result -- PASS" << std::endl;
  } else {
    std::cout << "  result -- FAIL" << std::endl;
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

