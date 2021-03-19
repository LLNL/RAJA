/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing the core components of RAJA::Teams
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_teams_core_HPP
#define RAJA_pattern_teams_core_HPP

#include "RAJA/config.hpp"
#include "RAJA/internal/get_platform.hpp"
#include "RAJA/util/StaticLayout.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/plugins.hpp"
#include "RAJA/util/types.hpp"
#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

#if defined(RAJA_DEVICE_CODE)
#define RAJA_TEAM_SHARED __shared__
#else
#define RAJA_TEAM_SHARED
#endif

namespace RAJA
{

namespace expt
{

// GPU or CPU threads available
enum ExecPlace { HOST, DEVICE, NUM_PLACES };

struct null_launch_t {
};

// Support for host, and device
template <typename HOST_POLICY
#if defined(RAJA_DEVICE_ACTIVE)
          ,
          typename DEVICE_POLICY
#endif
          >
struct LoopPolicy {
  using host_policy_t = HOST_POLICY;
#if defined(RAJA_DEVICE_ACTIVE)
  using device_policy_t = DEVICE_POLICY;
#endif
};

template <typename HOST_POLICY
#if defined(RAJA_DEVICE_ACTIVE)
          ,
          typename DEVICE_POLICY
#endif
          >
struct LaunchPolicy {
  using host_policy_t = HOST_POLICY;
#if defined(RAJA_DEVICE_ACTIVE)
  using device_policy_t = DEVICE_POLICY;
#endif
};


struct Teams {
  int value[3];

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr Teams() : value{1, 1, 1} {}

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr Teams(int i) : value{i, 1, 1} {}

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr Teams(int i, int j) : value{i, j, 1} {}

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr Teams(int i, int j, int k) : value{i, j, k} {}
};

struct Threads {
  int value[3];

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr Threads() : value{1, 1, 1} {}


  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr Threads(int i) : value{i, 1, 1} {}

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr Threads(int i, int j) : value{i, j, 1} {}

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr Threads(int i, int j, int k) : value{i, j, k} {}
};

struct Lanes {
  int value;

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr Lanes() : value(0) {}

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr Lanes(int i) : value(i) {}
};

struct Resources {
public:
  Teams teams;
  Threads threads;
  Lanes lanes;

  RAJA_INLINE
  Resources() = default;

  Resources(Teams in_teams, Threads in_threads)
      : teams(in_teams), threads(in_threads){};

private:
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Teams apply(Teams const &a) { return (teams = a); }

  RAJA_HOST_DEVICE
  RAJA_INLINE
  Threads apply(Threads const &a) { return (threads = a); }

  RAJA_HOST_DEVICE
  RAJA_INLINE
  Lanes apply(Lanes const &a) { return (lanes = a); }
};


class LaunchContext : public Resources
{
public:
  ExecPlace exec_place;

  LaunchContext(Resources const &base, ExecPlace place)
      : Resources(base), exec_place(place)
  {
  }


  RAJA_HOST_DEVICE
  void teamSync()
  {
#if defined(RAJA_DEVICE_CODE)
    __syncthreads();
#endif
  }
};


template <typename LAUNCH_POLICY>
struct LaunchExecute;

template <typename POLICY_LIST, typename BODY>
void launch(ExecPlace place, Resources const &team_resources, BODY const &body)
{
  switch (place) {
    case HOST: {
      using launch_t = LaunchExecute<typename POLICY_LIST::host_policy_t>;
      launch_t::exec(LaunchContext(team_resources, HOST), body);
      break;
    }
#ifdef RAJA_DEVICE_ACTIVE
    case DEVICE: {
      using launch_t = LaunchExecute<typename POLICY_LIST::device_policy_t>;
      launch_t::exec(LaunchContext(team_resources, DEVICE), body);
      break;
    }
#endif
    default:
      RAJA_ABORT_OR_THROW("Unknown launch place or device is not enabled");
  }
}


template<typename POLICY_LIST>
#if defined(RAJA_DEVICE_CODE)
using loop_policy = typename POLICY_LIST::device_policy_t;
#else
using loop_policy = typename POLICY_LIST::host_policy_t;
#endif

template <typename POLICY, typename SEGMENT>
struct LoopExecute;

template <typename POLICY, typename SEGMENT>
struct LoopICountExecute;

template <typename POLICY_LIST,
          typename CONTEXT,
          typename SEGMENT,
          typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void loop(CONTEXT const &ctx,
                                       SEGMENT const &segment,
                                       BODY const &body)
{

  LoopExecute<loop_policy<POLICY_LIST>, SEGMENT>::exec(ctx,
                                                       segment,
                                                       body);
}

template <typename POLICY_LIST,
          typename CONTEXT,
          typename SEGMENT,
          typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void loop_icount(CONTEXT const &ctx,
                                          SEGMENT const &segment,
                                          BODY const &body)
{

  LoopICountExecute<loop_policy<POLICY_LIST>, SEGMENT>::exec(ctx,
                                                          segment,
                                                          body);
}

template <typename POLICY_LIST,
          typename CONTEXT,
          typename SEGMENT,
          typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void loop(CONTEXT const &ctx,
                                       SEGMENT const &segment0,
                                       SEGMENT const &segment1,
                                       BODY const &body)
{

  LoopExecute<loop_policy<POLICY_LIST>, SEGMENT>::exec(ctx,
                                                       segment0,
                                                       segment1,
                                                       body);
}

template <typename POLICY_LIST,
          typename CONTEXT,
          typename SEGMENT,
          typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void loop_icount(CONTEXT const &ctx,
                                       SEGMENT const &segment0,
                                       SEGMENT const &segment1,
                                       SEGMENT const &segment2,
                                       BODY const &body)
{

  LoopICountExecute<loop_policy<POLICY_LIST>, SEGMENT>::exec(ctx,
                           segment0, segment1, segment2, body);
}




template <typename POLICY, typename SEGMENT>
struct TileExecute;

template <typename POLICY, typename SEGMENT>
struct TileICountExecute;

template <typename POLICY_LIST,
          typename CONTEXT,
          typename TILE_T,
          typename SEGMENT,
          typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void tile(CONTEXT const &ctx,
                                       TILE_T tile_size,
                                       SEGMENT const &segment,
                                       BODY const &body)
{

  TileExecute<loop_policy<POLICY_LIST>, SEGMENT>::exec(ctx,
                                                       tile_size,
                                                       segment,
                                                       body);
}

template <typename POLICY_LIST,
          typename CONTEXT,
          typename TILE_T,
          typename SEGMENT,
          typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void tile_icount(CONTEXT const &ctx,
                                       TILE_T tile_size,
                                       SEGMENT const &segment,
                                       BODY const &body)
{
  TileICountExecute<loop_policy<POLICY_LIST>, SEGMENT>::exec(ctx,
                                                          tile_size,
                                                          segment,
                                                          body);
}

template <typename POLICY_LIST,
          typename CONTEXT,
          typename TILE_T,
          typename SEGMENT,
          typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void tile(CONTEXT const &ctx,
                                       TILE_T tile_size0,
                                       TILE_T tile_size1,
                                       SEGMENT const &segment0,
                                       SEGMENT const &segment1,
                                       BODY const &body)
{

  TileExecute<loop_policy<POLICY_LIST>, SEGMENT>::exec(ctx,
                                                       tile_size0,
                                                       tile_size1,
                                                       segment0,
                                                       segment1,
                                                       body);
}

template <typename POLICY_LIST,
          typename CONTEXT,
          typename TILE_T,
          typename SEGMENT,
          typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void tile_icount(CONTEXT const &ctx,
                                       TILE_T tile_size0,
                                       TILE_T tile_size1,
                                       SEGMENT const &segment0,
                                       SEGMENT const &segment1,
                                       BODY const &body)
{

  TileICountExecute<loop_policy<POLICY_LIST>, SEGMENT>::exec(ctx,
                                                          tile_size0,
                                                          tile_size1,
                                                          segment0,
                                                          segment1,
                                                          body);
}

}  // namespace expt

}  // namespace RAJA
#endif
