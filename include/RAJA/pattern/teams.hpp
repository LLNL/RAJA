/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing user interface for RAJA::kernel
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_teams_HPP
#define RAJA_pattern_teams_HPP

#include "RAJA/config.hpp"

#include "RAJA/internal/get_platform.hpp"
#include "RAJA/util/plugins.hpp"

#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/loop/policy.hpp"
#include "RAJA/policy/cuda/policy.hpp"

namespace RAJA
{

enum ExecPlace { HOST, DEVICE };

template <ExecPlace EXEC_PLACE, typename POLICY>
struct LPolicy {
  static constexpr ExecPlace exec_place = EXEC_PLACE;
  using policy_t = POLICY;
};

struct Teams {
  int value[3];

  Teams() : value{1, 1, 1} {}

  Teams(int i) : value{i, 1, 1} {}

  Teams(int i, int j) : value{i, j, 1} {}

  Teams(int i, int j, int k) : value{i, j, k} {}
};

struct Threads {
  int value[3];

  Threads() : value{1, 1, 1} {}

  Threads(int i) : value{i, 1, 1} {}

  Threads(int i, int j) : value{i, j, 1} {}

  Threads(int i, int j, int k) : value{i, j, k} {}
};

struct Lanes {
  int value;

  Lanes() : value(0) {}

  Lanes(int i) : value(i) {}
};

class ResourceBase
{
public:
  Teams teams;
  Threads threads;
  Lanes lanes;
};

class LaunchContext : public ResourceBase
{
public:
  ExecPlace exec_place;

  LaunchContext(ResourceBase const &base, ExecPlace place)
      : ResourceBase(base), exec_place(place)
  {
  }
};

template <ExecPlace EXEC_PLACE>
class Resources : public ResourceBase
{
public:
  static constexpr ExecPlace exec_place = EXEC_PLACE;

  Resources() : ResourceBase() {}

  template <typename... ARGS>
  explicit Resources(ARGS const &... args) : ResourceBase()
  {
    camp::sink(apply(args)...);
  }

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

template <typename RESOURCE>
struct LaunchPlaceSwitchboard;

template <>
struct LaunchPlaceSwitchboard<Resources<HOST>> {
  template <typename BODY>
  static void exec(ExecPlace place, LaunchContext const &ctx, BODY const &body)
  {
    body(ctx);
  }
};


template <typename BODY>
__launch_bounds__(128, 1) __global__
    void launch_global_fcn(LaunchContext ctx, BODY body)
{
  // printf("Entering global function\n");
  body(ctx);
  // printf("Leaving global function\n");
}

template <>
struct LaunchPlaceSwitchboard<Resources<DEVICE>> {
  template <typename BODY>
  static void exec(ExecPlace place, LaunchContext const &ctx, BODY const &body)
  {
    // printf("Not implement yet!\n");

    dim3 blocks;
    dim3 threads;

    blocks.x = ctx.teams.value[0];
    blocks.y = ctx.teams.value[1];
    blocks.z = ctx.teams.value[2];

    threads.x = ctx.threads.value[0];
    threads.y = ctx.threads.value[1];
    threads.z = ctx.threads.value[2];
    launch_global_fcn<<<blocks, threads>>>(ctx, body);
    cudaDeviceSynchronize();
  }
};

template <typename RESOURCE_TUPLE, camp::idx_t I, camp::idx_t IMAX>
struct LaunchPlaceExtractor {

  template <typename BODY>
  static void launch(ExecPlace place,
                     RESOURCE_TUPLE const &resources,
                     BODY const &body)
  {

    using resource_t = camp::at_v<typename RESOURCE_TUPLE::TList, I>;

    if (place == resource_t::exec_place) {
      auto const &resource = camp::get<I>(resources);

      LaunchContext ctx(resource, place);

      LaunchPlaceSwitchboard<resource_t>::exec(place, ctx, body);
    } else {

      LaunchPlaceExtractor<RESOURCE_TUPLE, I + 1, IMAX>::launch(place,
                                                                resources,
                                                                body);
    }
  }
};


template <typename RESOURCE_TUPLE, camp::idx_t IMAX>
struct LaunchPlaceExtractor<RESOURCE_TUPLE, IMAX, IMAX> {
  template <typename BODY>
  static void launch(ExecPlace place,
                     RESOURCE_TUPLE const &resources,
                     BODY const &body)
  {
    printf("Failed to find resource requirements for execution place %d\n",
           (int)place);
  }
};

template <typename RESOURCES, typename BODY>
void launch(ExecPlace place, RESOURCES const &resources, BODY const &body)
{
  LaunchPlaceExtractor<
      RESOURCES,
      0,
      camp::size<typename RESOURCES::TList>::value>::launch(place,
                                                            resources,
                                                            body);
}


template <typename POLICY, typename SEGMENT>
struct LoopExecute;

template <typename SEGMENT>
struct LoopExecute<loop_exec, SEGMENT> {

  template <typename BODY>
  static RAJA_HOST_DEVICE void exec(LaunchContext const &ctx,
                                    SEGMENT const &segment,
                                    BODY const &body)
  {

    // block stride loop
    int len = segment.end() - segment.begin();
    for (int i = 0; i < len; i++) {

      body(*(segment.begin() + i));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<cuda_thread_x_loop, SEGMENT> {

  template <typename BODY>
  static RAJA_DEVICE void exec(LaunchContext const &ctx,
                               SEGMENT const &segment,
                               BODY const &body)
  {

    int len = segment.end() - segment.begin();

    for (int i = threadIdx.x; i < len; i += blockDim.x) {
      body(*(segment.begin() + i));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<cuda_thread_y_loop, SEGMENT> {

  template <typename BODY>
  static RAJA_DEVICE void exec(LaunchContext const &ctx,
                               SEGMENT const &segment,
                               BODY const &body)
  {

    int len = segment.end() - segment.begin();

    for (int i = threadIdx.y; i < len; i += blockDim.y) {
      body(*(segment.begin() + i));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<cuda_thread_z_loop, SEGMENT> {

  template <typename BODY>
  static RAJA_DEVICE void exec(LaunchContext const &ctx,
                               SEGMENT const &segment,
                               BODY const &body)
  {

    int len = segment.end() - segment.begin();

    for (int i = threadIdx.z; i < len; i += blockDim.z) {
      body(*(segment.begin() + i));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<cuda_block_x_loop, SEGMENT> {

  template <typename BODY>
  static RAJA_DEVICE void exec(LaunchContext const &ctx,
                               SEGMENT const &segment,
                               BODY const &body)
  {

    int len = segment.end() - segment.begin();

    for (int i = blockIdx.x; i < len; i += gridDim.x) {
      body(*(segment.begin() + i));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<cuda_block_x_direct, SEGMENT> {

  template <typename BODY>
  static RAJA_DEVICE void exec(LaunchContext const &ctx,
                               SEGMENT const &segment,
                               BODY const &body)
  {

    int len = segment.end() - segment.begin();
    {
      const int i = blockIdx.x;
      body(*(segment.begin() + i));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<cuda_block_y_direct, SEGMENT> {

  template <typename BODY>
  static RAJA_DEVICE void exec(LaunchContext const &ctx,
                               SEGMENT const &segment,
                               BODY const &body)
  {

    int len = segment.end() - segment.begin();
    {
      const int i = blockIdx.y;
      body(*(segment.begin() + i));
    }
  }
};


template <typename POLICY_LIST, camp::idx_t IDX, camp::idx_t MAX_IDX>
struct LoopPlaceSwitchboard {
  template <typename SEGMENT, typename BODY>
  static RAJA_HOST_DEVICE void exec(LaunchContext const &ctx,
                                    SEGMENT const &segment,
                                    BODY const &body)
  {
    if (camp::at_v<POLICY_LIST, IDX>::exec_place == ctx.exec_place) {
      LoopExecute<typename camp::at_v<POLICY_LIST, IDX>::policy_t,
                  SEGMENT>::exec(ctx, segment, body);
    } else {
      LoopPlaceSwitchboard<POLICY_LIST, IDX + 1, MAX_IDX>::exec(ctx,
                                                                segment,
                                                                body);
    }
  }
};

template <typename POLICY_LIST, camp::idx_t MAX_IDX>
struct LoopPlaceSwitchboard<POLICY_LIST, MAX_IDX, MAX_IDX> {
  template <typename SEGMENT, typename BODY>
  static RAJA_HOST_DEVICE void exec(LaunchContext const &ctx,
                                    SEGMENT const &segment,
                                    BODY const &body)
  {
    printf("whoops!");
  }
};


template <typename POLICY_LIST, typename SEGMENT, typename BODY>
RAJA_HOST_DEVICE void loop(LaunchContext const &ctx,
                           SEGMENT const &seg,
                           BODY const &body)
{


  LoopPlaceSwitchboard<POLICY_LIST, 0, camp::size<POLICY_LIST>::value>::exec(
      ctx, seg, body);
}


}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
