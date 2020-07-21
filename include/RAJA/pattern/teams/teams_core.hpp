/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing user interface for RAJA::Teams
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_teams_core_HPP
#define RAJA_pattern_teams_core_HPP

#include "RAJA/config.hpp"
#include "RAJA/internal/get_platform.hpp"
#include "RAJA/policy/cuda/policy.hpp"
#include "RAJA/policy/loop/policy.hpp"
#include "RAJA/policy/openmp/policy.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/plugins.hpp"
#include "RAJA/util/types.hpp"
#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"


#if defined(__CUDA_ARCH__)
#define TEAM_SHARED __shared__
#else
#define TEAM_SHARED
#endif

namespace RAJA
{

// GPU or CPU threads available
#if defined(RAJA_ENABLE_CUDA) && defined(RAJA_ENABLE_OPENMP)
enum ExecPlace { HOST, HOST_THREADS, DEVICE, NUM_PLACES };
#elif defined(RAJA_ENABLE_CUDA)
enum ExecPlace { HOST, DEVICE, NUM_PLACES };
#elif defined(RAJA_ENABLE_OPENMP)
enum ExecPlace { HOST, HOST_THREADS, NUM_PLACES };
#else
enum ExecPlace { HOST, NUM_PLACES };
#endif

// Support for Host, Host_threads, and Device
#if defined(RAJA_ENABLE_CUDA) && defined(RAJA_ENABLE_OPENMP)
template <typename HOST_POLICY,
          typename HOST_THREADS_POLICY,
          typename DEVICE_POLICY>
struct LoopPolicy {
  using host_policy_t = HOST_POLICY;
  using host_threads_policy_t = HOST_THREADS_POLICY;
  using device_policy_t = DEVICE_POLICY;
};

template <typename HOST_POLICY,
          typename HOST_THREADS_POLICY,
          typename DEVICE_POLICY>
struct LaunchPolicy {
  using host_policy_t = HOST_POLICY;
  using host_threads_policy_t = HOST_THREADS_POLICY;
  using device_policy_t = DEVICE_POLICY;
};
#elif defined(RAJA_ENABLE_CUDA)
template <typename HOST_POLICY, typename DEVICE_POLICY>
struct LoopPolicy {
  using host_policy_t = HOST_POLICY;
  using device_policy_t = DEVICE_POLICY;
};

template <typename HOST_POLICY, typename DEVICE_POLICY>
struct LaunchPolicy {
  using host_policy_t = HOST_POLICY;
  using device_policy_t = DEVICE_POLICY;
};
#elif defined(RAJA_ENABLE_OPENMP)
template <typename HOST_POLICY, typename HOST_THREADS_POLICY>
struct LoopPolicy {
  using host_policy_t = HOST_POLICY;
  using host_threads_policy_t = HOST_THREADS_POLICY;
};

template <typename HOST_POLICY, typename HOST_THREADS_POLICY>
struct LaunchPolicy {
  using host_policy_t = HOST_POLICY;
  using host_threads_policy_t = HOST_THREADS_POLICY;
};
#else
template <typename HOST_POLICY>
struct LoopPolicy {
  using host_policy_t = HOST_POLICY;
};

template <typename HOST_POLICY>
struct LaunchPolicy {
  using host_policy_t = HOST_POLICY;
};
#endif

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

  /*
  template <typename... ARGS>
  RAJA_INLINE
  explicit Resources(ARGS const &... args)
  {
    camp::sink(apply(args)...);
  }
  */
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

// Consolidate to one set of resources?
struct ResourceList {
  Resources team_resources;
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
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
  }
};

template <typename LAUNCH_POLICY>
struct LaunchExecute;

template <typename POLICY_LIST, typename BODY>
void launch(ExecPlace place, ResourceList const &resources, BODY const &body)
{

  if (place == HOST) {
    using launch_t = LaunchExecute<typename POLICY_LIST::host_policy_t>;

    launch_t::exec(LaunchContext(resources.team_resources, HOST), body);
  }
#ifdef RAJA_ENABLE_OPENMP
  else if (place == HOST_THREADS) {
    printf("Launching OMP code ! \n");
    using launch_t = LaunchExecute<typename POLICY_LIST::host_threads_policy_t>;
    launch_t::exec(LaunchContext(resources.team_resources, HOST_THREADS), body);
  }
#endif
#ifdef RAJA_ENABLE_CUDA
  else if (place == DEVICE) {
    using launch_t = LaunchExecute<typename POLICY_LIST::device_policy_t>;

    launch_t::exec(LaunchContext(resources.team_resources, DEVICE), body);
  }
#endif
  else {
    throw "unknown launch place!";
  }
}

template <typename POLICY, typename SEGMENT>
struct LoopExecute;


template <typename POLICY_LIST,
          typename CONTEXT,
          typename SEGMENT,
          typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void loop(CONTEXT const &ctx,
                                       SEGMENT const &segment,
                                       BODY const &body)
{
#ifdef __CUDA_ARCH__
  LoopExecute<typename POLICY_LIST::device_policy_t, SEGMENT>::exec(ctx,
                                                                    segment,
                                                                    body);
#else
  switch (ctx.exec_place) {
#if defined(RAJA_ENABLE_OPENMP)
    case HOST_THREADS:
      LoopExecute<typename POLICY_LIST::host_threads_policy_t, SEGMENT>::exec(
          ctx, segment, body);
      break;
#endif

    case HOST:
      LoopExecute<typename POLICY_LIST::host_policy_t, SEGMENT>::exec(ctx,
                                                                      segment,
                                                                      body);
      break;

    default:
      RAJA_ABORT_OR_THROW("Backend not support \n");
      break;
  }
#endif
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
#ifdef __CUDA_ARCH__
  LoopExecute<typename POLICY_LIST::device_policy_t, SEGMENT>::exec(ctx,
                                                                    segment0,
                                                                    segment1,
                                                                    body);
#else
  switch (ctx.exec_place) {
#if defined(RAJA_ENABLE_OPENMP)
    case HOST_THREADS:
      LoopExecute<typename POLICY_LIST::host_threads_policy_t, SEGMENT>::exec(
          ctx, segment0, segment1, body);
      break;
#endif

    case HOST:
      LoopExecute<typename POLICY_LIST::host_policy_t, SEGMENT>::exec(ctx,
                                                                      segment0,
                                                                      segment1,
                                                                      body);
      break;
      
  default: RAJA_ABORT_OR_THROW("Backend not support \n"); break;
  }
#endif
}

template <typename POLICY_LIST,
          typename CONTEXT,
          typename SEGMENT,
          typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void loop(CONTEXT const &ctx,
                                       SEGMENT const &segment0,
                                       SEGMENT const &segment1,
                                       SEGMENT const &segment2,
                                       BODY const &body)
{

#ifdef __CUDA_ARCH__
  LoopExecute<typename POLICY_LIST::device_policy_t, SEGMENT>::exec(
      ctx, segment0, segment1, segment2, body);
#else
  switch (ctx.exec_place) {

#ifdef RAJA_ENABLE_OPENMP
    case HOST_THREADS:
      LoopExecute<typename POLICY_LIST::host_threads_policy_t, SEGMENT>::exec(
          ctx, segment0, segment1, segment2, body);
      break;
#endif

    case HOST:
      LoopExecute<typename POLICY_LIST::host_policy_t, SEGMENT>::exec(
          ctx, segment0, segment1, segment2, body);
      break;

  default: RAJA_ABORT_OR_THROW("Backend not support \n"); break;
  }
#endif
}


}  // namespace RAJA
#endif
