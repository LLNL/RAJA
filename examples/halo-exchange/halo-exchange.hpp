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
#include <limits>

#include "RAJA/util/Timer.hpp"


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


struct TimerStats
{
  void start()
  {
    m_timer.start();
  }

  void stop()
  {
    m_timer.stop();
    RAJA::Timer::ElapsedType tCycle = m_timer.elapsed();
    m_timer.reset();

    m_num += 1u;
    m_tot += tCycle;
    if (tCycle < m_min) m_min = tCycle;
    if (tCycle > m_max) m_max = tCycle;
  }

  void reset()
  {
    m_timer.reset();
    m_num = 0u;
    m_tot = 0.0;
    m_min = std::numeric_limits<double>::max();
    m_max = std::numeric_limits<double>::min();
  }

  size_t get_num() const
  {
    return m_num;
  }

  double get_avg() const
  {
    return (m_num > 0u) ? m_tot / m_num : 0.0;
  }

  double get_min() const
  {
    return (m_num > 0u) ? m_min : 0.0;
  }

  double get_max() const
  {
    return (m_num > 0u) ? m_max : 0.0;
  }

private:
  RAJA::Timer m_timer;
  size_t m_num = 0u;
  double m_tot = 0.0;
  double m_min = std::numeric_limits<double>::max();
  double m_max = std::numeric_limits<double>::min();
};

#endif // RAJA_EXAMPLES_HALOEXCHANGE_HALOEXCHANGE_HPP
