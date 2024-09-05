/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA Fault Tolerance macros.
 *          RAJA Fault Tolerance only works when all the lambda
 *          functions passed to RAJA are in idempotent form,
 *          meaning there are no persistent variables in the
 *          lambda that have read-write semantics. In other words,
 *          persistent lambda function variables must be consistently
 *          used as read-only or write-only within the lambda scope.
 *
 *          These macros are designed to cooperate with an external
 *          signal handler that sets a global variable, fault_type,
 *          when a fault occurs. fault_type must be initialized to zero.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_fault_tolerance_HPP
#define RAJA_fault_tolerance_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_FT)

#if defined(RAJA_REPORT_FT)

#include <stdio.h>
#include "cycle.h"

#define RAJA_FT_BEGIN                                                          \
  extern volatile int fault_type;                                              \
  bool repeat;                                                                 \
  bool do_time = false;                                                        \
  ticks start = 0, stop = 0;                                                   \
  if (fault_type != 0)                                                         \
  {                                                                            \
    printf("Uncaught fault %d\n", fault_type);                                 \
    fault_type = 0;                                                            \
  }                                                                            \
  do                                                                           \
  {                                                                            \
    repeat = false;                                                            \
    if (do_time)                                                               \
    {                                                                          \
      start = getticks();                                                      \
    }

#define RAJA_FT_END                                                            \
  if (do_time)                                                                 \
  {                                                                            \
    stop = getticks();                                                         \
    printf("recoverable fault clock cycles = %16f\n", elapsed(stop, start));   \
    do_time = false;                                                           \
    fault_type = 0;                                                            \
  }                                                                            \
  if (fault_type < 0)                                                          \
  {                                                                            \
    printf("Unrecoverable fault (restart penalty)\n");                         \
    fault_type = 0;                                                            \
  }                                                                            \
  if (fault_type > 0)                                                          \
  {                                                                            \
    /* invalidate cache */                                                     \
    repeat = true;                                                             \
    do_time = true;                                                            \
  }                                                                            \
  }                                                                            \
  while (repeat == true)                                                       \
    ;

#else
#define RAJA_FT_BEGIN                                                          \
  extern volatile int fault_type;                                              \
  bool repeat;                                                                 \
  if (fault_type == 0)                                                         \
  {                                                                            \
    do                                                                         \
    {                                                                          \
      repeat = false;

#define RAJA_FT_END                                                            \
  if (fault_type > 0)                                                          \
  {                                                                            \
    /* invalidate cache */                                                     \
    repeat = true;                                                             \
    fault_type = 0;                                                            \
  }                                                                            \
  }                                                                            \
  while (repeat == true)                                                       \
    ;                                                                          \
  }                                                                            \
  else                                                                         \
  {                                                                            \
    fault_type = 0; /* ignore for the simulation */                            \
  }

#endif // RAJA_REPORT_FT

#else

#define RAJA_FT_BEGIN

#define RAJA_FT_END

#endif // RAJA_ENABLE_FT

#endif // closing endif for header file include guard
