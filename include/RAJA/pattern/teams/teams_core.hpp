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

#if defined(RAJA_DEVICE_CODE) && !defined(RAJA_ENABLE_SYCL)
RAJA_DEPRECATE("RAJA_TEAM_SHARED is not supported with SYCL, please use dyanmic shared mem")
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

struct Grid {
public:
  Teams teams;
  Threads threads;
  Lanes lanes;
  const char *kernel_name{nullptr};
  //size_t shared_mem_size; //In bytes

  RAJA_INLINE
  Grid() = default;

  Grid(Teams in_teams, Threads in_threads, const char *in_kernel_name = nullptr)
    : teams(in_teams), threads(in_threads), kernel_name(in_kernel_name){};

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

struct int3
{
  int x,y,z;
  int3(int _x=-1,int _y=-1,int _z=-1) :
    x(_x), y(_y), z(_z)
  {}
};

class LaunchContext : public Grid
{
public:

  //Will have to template on a type
  mutable size_t shared_mem_offset;
  mutable void *shared_mem_ptr; //pointer to dynamically allocated shared memory

#if defined(RAJA_ENABLE_SYCL)
  mutable int3 loc_id;
  mutable int3 group_id;
  mutable cl::sycl::nd_item<3> *itm;
#endif

  //int shared_mem; //how much shared memory is needed
  //shared_mem *my_sharedmem;

  LaunchContext(Grid const &base)
    : Grid(base), shared_mem_offset(0)
  {
  }

#if defined(RAJA_ENABLE_SYCL)
  template<typename T>
  T* getSharedMemory(size_t bytes)
  {
    T * mem_ptr = &((T*) shared_mem_ptr)[shared_mem_offset];
    shared_mem_offset += bytes*sizeof(T);

    //TODO add a check to ensure
    //we do not go beyond our allocated shared mem
    
    return mem_ptr;
  } 
  
#endif  

  RAJA_HOST_DEVICE
  void teamSync()
  {
#if defined(RAJA_ENABLE_SYCL) && defined(__SYCL_DEVICE_ONLY__)
    itm->barrier(sycl::access::fence_space::local_space);
#endif

#if defined(RAJA_DEVICE_CODE) && !defined(RAJA_ENABLE_SYCL)
    __syncthreads();
#endif
  }
};

template <typename LAUNCH_POLICY>
struct LaunchExecute;

//Policy based launch
template <typename LAUNCH_POLICY, typename BODY>
void launch(size_t shared_mem, Grid const &grid, BODY const &body)
{
  //Take the first policy as we assume the second policy is not user defined.
  //We rely on the user to pair launch and loop policies correctly.
  using launch_t = LaunchExecute<typename LAUNCH_POLICY::host_policy_t>;
  launch_t::exec(shared_mem, LaunchContext(grid), body);
}


//Run time based policy launch
template <typename POLICY_LIST, typename BODY>
void launch(size_t shared_mem,ExecPlace place, Grid const &grid, BODY const &body)
{
  switch (place) {
    case HOST: {
      using launch_t = LaunchExecute<typename POLICY_LIST::host_policy_t>;
      launch_t::exec(shared_mem, LaunchContext(grid), body);
      break;
    }
#ifdef RAJA_DEVICE_ACTIVE
    case DEVICE: {
      using launch_t = LaunchExecute<typename POLICY_LIST::device_policy_t>;
      launch_t::exec(shared_mem, LaunchContext(grid), body);
      break;
    }
#endif
    default:
      RAJA_ABORT_OR_THROW("Unknown launch place or device is not enabled");
  }
}

// Helper function to retrieve a resource based on the run-time policy - if a device is active
#if defined(RAJA_DEVICE_ACTIVE)
template<typename T, typename U>
RAJA::resources::Resource Get_Runtime_Resource(T host_res, U device_res, RAJA::expt::ExecPlace device){
  if(device == RAJA::expt::DEVICE) {return RAJA::resources::Resource(device_res);}
  else { return RAJA::resources::Resource(host_res); }
}
#else
template<typename T>
RAJA::resources::Resource Get_Host_Resource(T host_res, RAJA::expt::ExecPlace device){
  if(device == RAJA::expt::DEVICE) {RAJA_ABORT_OR_THROW("Device is not enabled");}

  return RAJA::resources::Resource(host_res);
}
#endif


//Launch API which takes team resource struct
template <typename POLICY_LIST, typename BODY>
resources::EventProxy<resources::Resource>
launch(RAJA::resources::Resource res, size_t shared_mem, Grid const &grid, BODY const &body)
{

  ExecPlace place;
  if(res.get_platform() == camp::resources::v1::Platform::host) {
    place = RAJA::expt::HOST;
  }else{
    place = RAJA::expt::DEVICE;
  }

  switch (place) {
    case HOST: {
      using launch_t = LaunchExecute<typename POLICY_LIST::host_policy_t>;
      return launch_t::exec(res, shared_mem, LaunchContext(grid), body); break;
    }
#ifdef RAJA_DEVICE_ACTIVE
    case DEVICE: {
      using launch_t = LaunchExecute<typename POLICY_LIST::device_policy_t>;
      return launch_t::exec(res, shared_mem, LaunchContext(grid), body); break;
    }
#endif
    default: {
      RAJA_ABORT_OR_THROW("Unknown launch place or device is not enabled");
    }
  }
  //Should not get here;
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
