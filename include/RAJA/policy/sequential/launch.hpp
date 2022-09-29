/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing user interface for RAJA::Teams::seq
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_launch_sequential_HPP
#define RAJA_pattern_launch_sequential_HPP

#include "RAJA/pattern/launch/launch_core.hpp"
#include "RAJA/policy/sequential/policy.hpp"


namespace RAJA
{

template <typename SEGMENT>
struct LoopExecute<seq_exec, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    RAJA_NO_SIMD
    for (int i = 0; i < len; i++) {
      body(*(segment.begin() + i));
    }
  }
};

template <typename SEGMENT>
struct LoopICountExecute<seq_exec, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    RAJA_NO_SIMD
    for (int i = 0; i < len; i++) {
      body(*(segment.begin() + i), i);
    }
  }
};

}  // namespace RAJA
#endif
