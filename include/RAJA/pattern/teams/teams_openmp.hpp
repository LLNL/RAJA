/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing user interface for RAJA::Teams::openmp
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_teams_openmp_HPP
#define RAJA_pattern_teams_openmp_HPP

#include "RAJA/pattern/teams/teams_core.hpp"
#include "RAJA/policy/openmp/policy.hpp"

namespace RAJA
{

namespace expt
{

struct omp_launch_t {
};

template <>
struct LaunchExecute<RAJA::expt::omp_launch_t> {
  template <typename BODY>
  static void exec(LaunchContext const &ctx, BODY const &body)
  {
    body(ctx);
  }
};


template <typename SEGMENT>
struct LoopExecute<omp_parallel_for_exec, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    int len = segment.end() - segment.begin();
#pragma omp parallel for
    for (int i = 0; i < len; i++) {

      body(*(segment.begin() + i));
    }
  }

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {

    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

#pragma omp parallel for RAJA_COLLAPSE(2)
    for (int j = 0; j < len1; j++) {
      for (int i = 0; i < len0; i++) {

        body(*(segment0.begin() + i), *(segment1.begin() + j));
      }
    }
  }

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {

    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

#pragma omp parallel for RAJA_COLLAPSE(3)
    for (int k = 0; k < len2; k++) {
      for (int j = 0; j < len1; j++) {
        for (int i = 0; i < len0; i++) {
          body(*(segment0.begin() + i),
               *(segment1.begin() + j),
               *(segment2.begin() + k));
        }
      }
    }
  }
};

  //
  // Return local index
  //
template <typename SEGMENT>
struct LoopIdxExecute<omp_parallel_for_exec, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    int len = segment.end() - segment.begin();
#pragma omp parallel for
    for (int i = 0; i < len; i++) {

      body(*(segment.begin() + i), i);
    }
  }

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {

    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

#pragma omp parallel for RAJA_COLLAPSE(2)
    for (int j = 0; j < len1; j++) {
      for (int i = 0; i < len0; i++) {

        body(*(segment0.begin() + i),
             *(segment1.begin() + j),
             i, j);
      }
    }
  }

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {

    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

#pragma omp parallel for RAJA_COLLAPSE(3)
    for (int k = 0; k < len2; k++) {
      for (int j = 0; j < len1; j++) {
        for (int i = 0; i < len0; i++) {
          body(*(segment0.begin() + i),
               *(segment1.begin() + j),
               *(segment2.begin() + k),
               i, j, k);
        }
      }
    }
  }
};

// policy for perfectly nested loops
struct omp_parallel_nested_for_exec;

template <typename SEGMENT>
struct LoopExecute<omp_parallel_nested_for_exec, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {

    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

#pragma omp parallel for RAJA_COLLAPSE(2)
    for (int j = 0; j < len1; j++) {
      for (int i = 0; i < len0; i++) {

        body(*(segment0.begin() + i), *(segment1.begin() + j));
      }
    }
  }

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {

    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

#pragma omp parallel for RAJA_COLLAPSE(3)
    for (int k = 0; k < len2; k++) {
      for (int j = 0; j < len1; j++) {
        for (int i = 0; i < len0; i++) {
          body(*(segment0.begin() + i),
               *(segment1.begin() + j),
               *(segment2.begin() + k));
        }
      }
    }
  }
};

//Return local index
template <typename SEGMENT>
struct LoopIdxExecute<omp_parallel_nested_for_exec, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {

    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

#pragma omp parallel for RAJA_COLLAPSE(2)
    for (int j = 0; j < len1; j++) {
      for (int i = 0; i < len0; i++) {

        body(*(segment0.begin() + i), *(segment1.begin() + j),
             i, j);
      }
    }
  }

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {

    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

#pragma omp parallel for RAJA_COLLAPSE(3)
    for (int k = 0; k < len2; k++) {
      for (int j = 0; j < len1; j++) {
        for (int i = 0; i < len0; i++) {
          body(*(segment0.begin() + i),
               *(segment1.begin() + j),
               *(segment2.begin() + k), i, j, k);
        }
      }
    }
  }
};


template <typename SEGMENT>
struct TileExecute<omp_parallel_for_exec, SEGMENT> {

  template <typename BODY, typename TILE_T>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    int len = segment.end() - segment.begin();
#pragma omp parallel for
    for (int i = 0; i < len; i+= tile_size) {

      body(segment.slice(i, tile_size));
    }
  }

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size0,
      TILE_T tile_size1,
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {

    const int len0 = segment0.end() - segment0.begin();
    const int len1 = segment1.end() - segment1.begin();

#pragma omp parallel for RAJA_COLLAPSE(2)
    for (int ty = 0; ty < len1; ty += tile_size1) {
      for (int tx = 0; tx < len0; tx += tile_size0) {
        body(segment0.slice(tx, tile_size0), segment1.slice(ty,tile_size1));
      }
    }

  }

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size0,
      TILE_T tile_size1,
      TILE_T tile_size2,
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {

    const int len0 = segment0.end() - segment0.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len2 = segment2.end() - segment2.begin();

#pragma omp parallel for RAJA_COLLAPSE(3)
    for (int tz = 0; tz < len2; tz += tile_size2) {
      for (int ty = 0; ty < len1; ty += tile_size1) {
        for (int tx = 0; tx < len0; tx += tile_size0) {
        body(segment0.slice(tx, tile_size0),
             segment1.slice(ty,tile_size1),
             segment1.slice(ty,tile_size2));
        }
      }
    }

  }

};

template <typename SEGMENT>
struct TileIdxExecute<omp_parallel_for_exec, SEGMENT> {

  template <typename BODY, typename TILE_T>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    int len = segment.end() - segment.begin();
#pragma omp parallel for
    for (int i = 0; i < len; i+= tile_size) {

      body(segment.slice(i, tile_size), i);
    }
  }

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size0,
      TILE_T tile_size1,
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {

    const int len0 = segment0.end() - segment0.begin();
    const int len1 = segment1.end() - segment1.begin();

#pragma omp parallel for RAJA_COLLAPSE(2)
    for (int ty = 0; ty < len1; ty += tile_size1) {
      for (int tx = 0; tx < len0; tx += tile_size0) {
        body(segment0.slice(tx, tile_size0), segment1.slice(ty,tile_size1), tx, ty);
      }
    }

  }


  template <typename TILE_T, typename BODY>
  static RAJA_INLINE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size0,
      TILE_T tile_size1,
      TILE_T tile_size2,
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {

    const int len0 = segment0.end() - segment0.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len2 = segment2.end() - segment2.begin();

#pragma omp parallel for RAJA_COLLAPSE(3)
    for (int tz = 0; tz < len2; tz += tile_size2) {
      for (int ty = 0; ty < len1; ty += tile_size1) {
        for (int tx = 0; tx < len0; tx += tile_size0) {
        body(segment0.slice(tx, tile_size0),
             segment1.slice(ty,tile_size1),
             segment1.slice(ty,tile_size2), tx, ty ,tz);
        }
      }
    }

  }

};

}  // namespace expt

}  // namespace RAJA
#endif
