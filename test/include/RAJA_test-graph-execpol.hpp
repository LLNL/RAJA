//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Execution policy lists used throughout graph tests
//

#ifndef __RAJA_test_graph_execpol_HPP__
#define __RAJA_test_graph_execpol_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/list.hpp"

// Sequential execution policy types
using SequentialGraphExecPols = camp::list< RAJA::seq_graph,
                                            RAJA::loop_graph >;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPGraphExecPols = camp::list<
#if defined(RAJA_ENABLE_OPENMP_TASK_DEPEND)
                                        RAJA::omp_task_graph
#else
                                        RAJA::loop_graph // must have at least one entry
#endif
                                       >;
#endif  // RAJA_ENABLE_OPENMP

#endif  // __RAJA_test_graph_execpol_HPP__
