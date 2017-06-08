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

#ifndef RAJA_fault_tolerance_HPP
#define RAJA_fault_tolerance_HPP

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/config.hpp"

#ifdef RAJA_ENABLE_FT

#ifdef RAJA_REPORT_FT
#include <stdio.h>
#include "cycle.h"

#define RAJA_FT_BEGIN                          \
  extern volatile int fault_type;              \
  bool repeat;                                 \
  bool do_time = false;                        \
  ticks start = 0, stop = 0;                   \
  if (fault_type != 0) {                       \
    printf("Uncaught fault %d\n", fault_type); \
    fault_type = 0;                            \
  }                                            \
  do {                                         \
    repeat = false;                            \
    if (do_time) {                             \
      start = getticks();                      \
    }

#define RAJA_FT_END                                                          \
  if (do_time) {                                                             \
    stop = getticks();                                                       \
    printf("recoverable fault clock cycles = %16f\n", elapsed(stop, start)); \
    do_time = false;                                                         \
    fault_type = 0;                                                          \
  }                                                                          \
  if (fault_type < 0) {                                                      \
    printf("Unrecoverable fault (restart penalty)\n");                       \
    fault_type = 0;                                                          \
  }                                                                          \
  if (fault_type > 0) {                                                      \
    /* invalidate cache */                                                   \
    repeat = true;                                                           \
    do_time = true;                                                          \
  }                                                                          \
  }                                                                          \
  while (repeat == true)                                                     \
    ;

#else
#define RAJA_FT_BEGIN             \
  extern volatile int fault_type; \
  bool repeat;                    \
  if (fault_type == 0) {          \
    do {                          \
      repeat = false;

#define RAJA_FT_END        \
  if (fault_type > 0) {    \
    /* invalidate cache */ \
    repeat = true;         \
    fault_type = 0;        \
  }                        \
  }                        \
  while (repeat == true)   \
    ;                      \
  }                        \
  else { fault_type = 0; /* ignore for the simulation */ }

#endif  // RAJA_REPORT_FT

#else

#define RAJA_FT_BEGIN

#define RAJA_FT_END

#endif  // RAJA_ENABLE_FT

#endif  // closing endif for header file include guard
