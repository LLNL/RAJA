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
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_launch_core_HPP
#define RAJA_pattern_launch_core_HPP

#include "RAJA/config.hpp"
#include "RAJA/internal/get_platform.hpp"
#include "RAJA/util/StaticLayout.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/plugins.hpp"
#include "RAJA/util/types.hpp"
#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

//Odd dependecy with atomics is breaking CI builds
//#include "RAJA/util/View.hpp"

#if defined(RAJA_DEVICE_CODE) && !defined(RAJA_ENABLE_SYCL)
#define RAJA_TEAM_SHARED __shared__
#else
#define RAJA_TEAM_SHARED
#endif

namespace RAJA
{

// GPU or CPU threads available
//strongly type the ExecPlace (guards agaist errors)
enum struct ExecPlace : int { HOST, DEVICE, NUM_PLACES };

struct null_launch_t {
};

// Support for host, and device
template <typename HOST_POLICY
#if defined(RAJA_DEVICE_ACTIVE)
          ,
          typename DEVICE_POLICY = HOST_POLICY
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
          typename DEVICE_POLICY = HOST_POLICY
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

struct LaunchParams {
public:
  Teams teams;
  Threads threads;
  size_t shared_mem_size;

  RAJA_INLINE
  LaunchParams() = default;

  LaunchParams(Teams in_teams, Threads in_threads, size_t in_shared_mem_size = 0)
    : teams(in_teams), threads(in_threads), shared_mem_size(in_shared_mem_size) {};

private:
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Teams apply(Teams const &a) { return (teams = a); }

  RAJA_HOST_DEVICE
  RAJA_INLINE
  Threads apply(Threads const &a) { return (threads = a); }
};

class LaunchContext
{
public:

  //Bump style allocator used to
  //get memory from the pool
  size_t shared_mem_offset;

  void *shared_mem_ptr;

#if defined(RAJA_ENABLE_SYCL)
  mutable cl::sycl::nd_item<3> *itm;
#endif

  RAJA_HOST_DEVICE LaunchContext()
    : shared_mem_offset(0), shared_mem_ptr(nullptr)
  {
  }

  //TODO handle alignment
  template<typename T>
  RAJA_HOST_DEVICE T* getSharedMemory(size_t bytes)
  {
    T * mem_ptr = &((T*) shared_mem_ptr)[shared_mem_offset];

    shared_mem_offset += bytes*sizeof(T);
    return mem_ptr;
  }

  /*
  //Odd dependecy with atomics is breaking CI builds
  template<typename T, size_t DIM, typename IDX_T=RAJA::Index_type, ptrdiff_t z_stride=DIM-1, typename arg, typename... args>
  RAJA_HOST_DEVICE auto getSharedMemoryView(size_t bytes, arg idx, args... idxs)
  {
    T * mem_ptr = &((T*) shared_mem_ptr)[shared_mem_offset];

    shared_mem_offset += bytes*sizeof(T);
    return RAJA::View<T, RAJA::Layout<DIM, IDX_T, z_stride>>(mem_ptr, idx, idxs...);
  }
  */

  RAJA_HOST_DEVICE void releaseSharedMemory()
  {
    //On the cpu/gpu we want to restart the count
    shared_mem_offset = 0;
  }

  RAJA_HOST_DEVICE
  void teamSync()
  {
#if defined(RAJA_DEVICE_CODE) && defined(RAJA_ENABLE_SYCL)
    itm->barrier(sycl::access::fence_space::local_space);
#endif

#if defined(RAJA_DEVICE_CODE) && !defined(RAJA_ENABLE_SYCL)
    __syncthreads();
#endif
  }
};

template <typename LAUNCH_POLICY>
struct LaunchExecute;

//Policy based launch without name argument
template <typename LAUNCH_POLICY, typename BODY>
void launch(LaunchParams const &params, BODY const &body)
{
  launch<LAUNCH_POLICY>(params, nullptr, body);
}

//Policy based launch
template <typename LAUNCH_POLICY, typename BODY>
void launch(LaunchParams const &params, const char *kernel_name, BODY const &body)
{
  //Take the first policy as we assume the second policy is not user defined.
  //We rely on the user to pair launch and loop policies correctly.
  util::PluginContext context{util::make_context<typename LAUNCH_POLICY::host_policy_t>()};
  util::callPreCapturePlugins(context);

  using RAJA::util::trigger_updates_before;
  auto p_body = trigger_updates_before(body);

  util::callPostCapturePlugins(context);

  util::callPreLaunchPlugins(context);

  using launch_t = LaunchExecute<typename LAUNCH_POLICY::host_policy_t>;
  launch_t::exec(params, kernel_name, p_body);

  util::callPostLaunchPlugins(context);
}


//Run time based policy launch
template <typename POLICY_LIST, typename BODY>
void launch(ExecPlace place, LaunchParams const &params, BODY const &body)
{
  launch<POLICY_LIST>(place, params, nullptr, body);
}

template <typename POLICY_LIST, typename BODY>
void launch(ExecPlace place, const LaunchParams &params, const char *kernel_name, BODY const &body)
{

  //Forward to single policy launch API - simplifies testing of plugins
  switch (place) {
    case ExecPlace::HOST: {
      launch<LaunchPolicy<typename POLICY_LIST::host_policy_t>>(params, kernel_name, body);
      break;
    }
#ifdef RAJA_DEVICE_ACTIVE
  case ExecPlace::DEVICE: {
      launch<LaunchPolicy<typename POLICY_LIST::device_policy_t>>(params, kernel_name, body);
      break;
    }
#endif
    default:
      RAJA_ABORT_OR_THROW("Unknown launch place or device is not enabled");
  }

}

// Helper function to retrieve a resource based on the run-time policy - if a device is active
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
template<typename T, typename U>
RAJA::resources::Resource Get_Runtime_Resource(T host_res, U device_res, RAJA::ExecPlace device){
  if(device == RAJA::ExecPlace::DEVICE) {return RAJA::resources::Resource(device_res);}
  else { return RAJA::resources::Resource(host_res); }
}
#else
template<typename T>
RAJA::resources::Resource Get_Host_Resource(T host_res, RAJA::ExecPlace device){
  if(device == RAJA::ExecPlace::DEVICE) {RAJA_ABORT_OR_THROW("Device is not enabled");}

  return RAJA::resources::Resource(host_res);
}
#endif


//Launch API which takes team resource struct
template <typename POLICY_LIST, typename BODY>
resources::EventProxy<resources::Resource>
launch(RAJA::resources::Resource res, LaunchParams const &params, BODY const &body)
{
  return launch<POLICY_LIST>(res, params, nullptr, body);
}

template <typename POLICY_LIST, typename BODY>
resources::EventProxy<resources::Resource>
launch(RAJA::resources::Resource res, LaunchParams const &params, const char *kernel_name, BODY const &body)
{

  ExecPlace place;
  if(res.get_platform() == RAJA::Platform::host) {
    place = RAJA::ExecPlace::HOST;
  }else{
    place = RAJA::ExecPlace::DEVICE;
  }

  //
  //Configure plugins
  //
#ifdef RAJA_DEVICE_ACTIVE
  util::PluginContext context{place == ExecPlace::HOST ?
      util::make_context<typename POLICY_LIST::host_policy_t>()
      : util::make_context<typename POLICY_LIST::device_policy_t>()};
#else
  util::PluginContext context{util::make_context<typename POLICY_LIST::host_policy_t>()};
#endif

  util::callPreCapturePlugins(context);

  using RAJA::util::trigger_updates_before;
  auto p_body = trigger_updates_before(body);

  util::callPostCapturePlugins(context);

  util::callPreLaunchPlugins(context);

  switch (place) {
    case ExecPlace::HOST: {
      using launch_t = LaunchExecute<typename POLICY_LIST::host_policy_t>;
      resources::EventProxy<resources::Resource> e_proxy = launch_t::exec(res, params, kernel_name, p_body);
      util::callPostLaunchPlugins(context);
      return e_proxy;
    }
#ifdef RAJA_DEVICE_ACTIVE
    case ExecPlace::DEVICE: {
      using launch_t = LaunchExecute<typename POLICY_LIST::device_policy_t>;
      resources::EventProxy<resources::Resource> e_proxy = launch_t::exec(res, params, kernel_name, p_body);
      util::callPostLaunchPlugins(context);
      return e_proxy;
    }
#endif
    default: {
      RAJA_ABORT_OR_THROW("Unknown launch place or device is not enabled");
    }
  }

  RAJA_ABORT_OR_THROW("Unknown launch place");

  //^^ RAJA will abort before getting here
  return resources::EventProxy<resources::Resource>(res);
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

RAJA_SUPPRESS_HD_WARN
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

namespace expt
{

RAJA_SUPPRESS_HD_WARN
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

RAJA_SUPPRESS_HD_WARN
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

} //namespace expt

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

namespace expt
{

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

} //namespace expt

}  // namespace RAJA
#endif
