/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for OpenMP
 *          OpenMP execution.
 *
 *          These methods should work on any platform that supports OpenMP.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
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

#ifndef RAJA_omp_target_reduce_HPP
#define RAJA_omp_target_reduce_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "RAJA/util/types.hpp"

#include "RAJA/pattern/reduce.hpp"

#include "RAJA/policy/openmp/policy.hpp"

#include <omp.h>
#include <algorithm>

namespace RAJA
{

namespace omp
{

//! Information necessary for OpenMP offload to be considered
struct Offload_Info {
  int hostID{omp_get_initial_device()};
  int deviceID{omp_get_default_device()};
  bool isMapped{false};

  Offload_Info() = default;

  Offload_Info(const Offload_Info &other)
      : hostID{other.hostID}, deviceID{other.deviceID}, isMapped{other.isMapped}
  {
  }
};

//! Reduction data for OpenMP Offload -- stores value, host pointer, and device
//! pointer
template <size_t Teams, typename T>
struct Reduce_Data {
  mutable T value;
  T *device;
  T *host;

  //! disallow default constructor
  Reduce_Data() = delete;

  /*! \brief create from a default value and offload information
   *
   *  allocates data on the host and device and initializes values to default
   */
  explicit Reduce_Data(T defaultValue, T identityValue, Offload_Info &info)
      : value{defaultValue},
        device{reinterpret_cast<T *>(
            omp_target_alloc(Teams * sizeof(T), info.deviceID))},
        host{new T[Teams]}
  {
    if (!host) {
      printf("Unable to allocate space on host\n");
      exit(1);
    }
    if (!device) {
      printf("Unable to allocate space on device\n");
      exit(1);
    }
    std::fill_n(host, Teams, identityValue);
    hostToDevice(info);
  }

  //! default copy constructor for POD
  Reduce_Data(const Reduce_Data &) = default;

  //! transfers from the host to the device -- exit() is called upon failure
  RAJA_INLINE void hostToDevice(Offload_Info &info)
  {
    // precondition: host and device are valid pointers
    if (omp_target_memcpy(reinterpret_cast<void *>(device),
                          reinterpret_cast<void *>(host),
                          Teams * sizeof(T),
                          0,
                          0,
                          info.deviceID,
                          info.hostID)
        != 0) {
      printf("Unable to copy memory from host to device\n");
      exit(1);
    }
  }

  //! transfers from the device to the host -- exit() is called upon failure
  RAJA_INLINE void deviceToHost(Offload_Info &info)
  {
    // precondition: host and device are valid pointers
    if (omp_target_memcpy(reinterpret_cast<void *>(host),
                          reinterpret_cast<void *>(device),
                          Teams * sizeof(T),
                          0,
                          0,
                          info.hostID,
                          info.deviceID)
        != 0) {
      printf("Unable to copy memory from device to host\n");
      exit(1);
    }
  }

  //! frees all data from the offload information passed
  RAJA_INLINE void cleanup(Offload_Info &info)
  {
    if (device) {
      omp_target_free(reinterpret_cast<void *>(device), info.deviceID);
      device = nullptr;
    }
    if (host) {
      delete[] host;
      host = nullptr;
    }
  }
};

}  // end namespace omp

//! OpenMP Target Reduction entity -- generalize on # of teams, reduction, and
//! type
template <size_t Teams, typename Reducer, typename T>
struct TargetReduce {
  TargetReduce() = delete;
  TargetReduce(const TargetReduce &) = default;

  explicit TargetReduce(T init_val)
      : info(), val(init_val, Reducer::identity, info)
  {
  }

  //! apply reduction on device upon destruction
  ~TargetReduce()
  {
    if (!omp_is_initial_device()) {
#pragma omp critical
      {
        int tid = omp_get_team_num();
        // printf("%d:%p\n", tid, val.device);
        Reducer{}(val.device[tid], val.value);
      }
    }
  }

  //! map result value back to host if not done already; return aggregate value
  operator T()
  {
    if (!info.isMapped) {
      val.deviceToHost(info);
      for (int i = 0; i < Teams; ++i) {
        Reducer{}(val.value, val.host[i]);
      }
      val.cleanup(info);
      info.isMapped = true;
    }
    return val.value;
  }
  //! alias for operator T()
  T get() { return operator T(); }

  //! apply reduction
  TargetReduce &reduce(T rhsVal)
  {
    Reducer{}(val.value, rhsVal);
    return *this;
  }

  //! apply reduction (const version) -- still reduces internal values
  const TargetReduce &reduce(T rhsVal) const
  {
    Reducer{}(val.value, rhsVal);
    return *this;
  }

private:
  //! storage for offload information (host ID, device ID)
  omp::Offload_Info info;
  //! storage for reduction data (host ptr, device ptr, value)
  omp::Reduce_Data<Teams, T> val;
};

//! OpenMP Target Reduction Location entity -- generalize on # of teams,
//! reduction, and type
template <size_t Teams, typename Reducer, typename T, typename IndexType>
struct TargetReduceLoc {
  TargetReduceLoc() = delete;
  TargetReduceLoc(const TargetReduceLoc &) = default;
  explicit TargetReduceLoc(T init_val, IndexType init_loc)
      : info(),
        val(init_val, Reducer::identity, info),
        loc(init_loc, IndexType(-1), info)
  {
  }

  //! apply reduction on device upon destruction
  ~TargetReduceLoc()
  {
    if (!omp_is_initial_device()) {
#pragma omp critical
      {
        int tid = omp_get_team_num();
        Reducer{}(val.device[tid], loc.device[tid], val.value, loc.value);
      }
    }
  }

  //! map result value back to host if not done already; return aggregate value
  operator T()
  {
    if (!info.isMapped) {
      val.deviceToHost(info);
      loc.deviceToHost(info);
      for (int i = 0; i < Teams; ++i) {
        Reducer{}(val.value, loc.value, val.host[i], loc.host[i]);
      }
      val.cleanup(info);
      loc.cleanup(info);
      info.isMapped = true;
    }
    return val.value;
  }
  //! alias for operator T()
  T get() { return operator T(); }

  //! map result value back to host if not done already; return aggregate
  //! location
  IndexType getLoc()
  {
    if (!info.isMapped) get();
    return loc.value;
  }

  //! apply reduction
  TargetReduceLoc &reduce(T rhsVal, IndexType rhsLoc)
  {
    Reducer{}(val.value, loc.value, rhsVal, rhsLoc);
    return *this;
  }

  //! apply reduction (const version) -- still reduces internal values
  const TargetReduceLoc &reduce(T rhsVal, IndexType rhsLoc) const
  {
    Reducer{}(val.value, loc.value, rhsVal, rhsLoc);
    return *this;
  }

private:
  //! storage for offload information
  omp::Offload_Info info;
  //! storage for reduction data for value
  omp::Reduce_Data<Teams, T> val;
  //! storage for redcution data for location
  omp::Reduce_Data<Teams, IndexType> loc;
};

//! specialization of ReduceSum for omp_target_reduce
template <size_t Teams, typename T>
struct ReduceSum<omp_target_reduce<Teams>, T>
    : public TargetReduce<Teams, RAJA::reduce::sum<T>, T> {
  using self = ReduceSum<omp_target_reduce<Teams>, T>;
  using parent = TargetReduce<Teams, RAJA::reduce::sum<T>, T>;
  using parent::parent;
  //! enable operator+= for ReduceSum -- alias for reduce()
  self &operator+=(T rhsVal)
  {
    parent::reduce(rhsVal);
    return *this;
  }
  //! enable operator+= for ReduceSum -- alias for reduce()
  const self &operator+=(T rhsVal) const
  {
    parent::reduce(rhsVal);
    return *this;
  }
};

//! specialization of ReduceMin for omp_target_reduce
template <size_t Teams, typename T>
struct ReduceMin<omp_target_reduce<Teams>, T>
    : public TargetReduce<Teams, RAJA::reduce::min<T>, T> {
  using self = ReduceMin<omp_target_reduce<Teams>, T>;
  using parent = TargetReduce<Teams, RAJA::reduce::min<T>, T>;
  using parent::parent;
  //! enable min() for ReduceMin -- alias for reduce()
  self &min(T rhsVal)
  {
    parent::reduce(rhsVal);
    return *this;
  }
  //! enable min() for ReduceMin -- alias for reduce()
  const self &min(T rhsVal) const
  {
    parent::reduce(rhsVal);
    return *this;
  }
};

//! specialization of ReduceMax for omp_target_reduce
template <size_t Teams, typename T>
struct ReduceMax<omp_target_reduce<Teams>, T>
    : public TargetReduce<Teams, RAJA::reduce::max<T>, T> {
  using self = ReduceMax<omp_target_reduce<Teams>, T>;
  using parent = TargetReduce<Teams, RAJA::reduce::max<T>, T>;
  using parent::parent;
  //! enable max() for ReduceMax -- alias for reduce()
  self &max(T rhsVal)
  {
    parent::reduce(rhsVal);
    return *this;
  }
  //! enable max() for ReduceMax -- alias for reduce()
  const self &max(T rhsVal) const
  {
    parent::reduce(rhsVal);
    return *this;
  }
};

//! specialization of ReduceMinLoc for omp_target_reduce
template <size_t Teams, typename T>
struct ReduceMinLoc<omp_target_reduce<Teams>, T>
    : public TargetReduceLoc<Teams,
                             RAJA::reduce::minloc<T, Index_type>,
                             T,
                             Index_type> {
  using self = ReduceMinLoc<omp_target_reduce<Teams>, T>;
  using parent = TargetReduceLoc<Teams,
                                 RAJA::reduce::minloc<T, Index_type>,
                                 T,
                                 Index_type>;
  using parent::parent;
  //! enable minloc() for ReduceMinLoc -- alias for reduce()
  self &minloc(T rhsVal, Index_type rhsLoc)
  {
    parent::reduce(rhsVal, rhsLoc);
    return *this;
  }
  //! enable minloc() for ReduceMinLoc -- alias for reduce()
  const self &minloc(T rhsVal, Index_type rhsLoc) const
  {
    parent::reduce(rhsVal, rhsLoc);
    return *this;
  }
};

//! specialization of ReduceMaxLoc for omp_target_reduce
template <size_t Teams, typename T>
struct ReduceMaxLoc<omp_target_reduce<Teams>, T>
    : public TargetReduceLoc<Teams,
                             RAJA::reduce::maxloc<T, Index_type>,
                             T,
                             Index_type> {
  using self = ReduceMaxLoc<omp_target_reduce<Teams>, T>;
  using parent = TargetReduceLoc<Teams,
                                 RAJA::reduce::maxloc<T, Index_type>,
                                 T,
                                 Index_type>;
  using parent::parent;
  //! enable maxloc() for ReduceMaxLoc -- alias for reduce()
  self &maxloc(T rhsVal, Index_type rhsLoc)
  {
    parent::reduce(rhsVal, rhsLoc);
    return *this;
  }
  //! enable maxloc() for ReduceMaxLoc -- alias for reduce()
  const self &maxloc(T rhsVal, Index_type rhsLoc) const
  {
    parent::reduce(rhsVal, rhsLoc);
    return *this;
  }
};

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_TARGET_OPENMP guard

#endif  // closing endif for header file include guard
