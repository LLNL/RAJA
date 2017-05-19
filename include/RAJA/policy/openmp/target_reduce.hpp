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

#ifndef RAJA_target_reduce_omp_HXX
#define RAJA_target_reduce_omp_HXX

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

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

#include "RAJA/util/types.hpp"

#include "RAJA/pattern/reduce.hpp"

#include "RAJA/policy/openmp/policy.hpp"

#include <omp.h>
#include <algorithm>

namespace RAJA
{

namespace omp
{

template <typename Self>
struct Offload_Info {
  int hostID{omp_get_initial_device()};
  int deviceID{omp_get_default_device()};
  bool isMapped{false};

  Offload_Info() = default;

  Offload_Info(const Offload_Info &other)
      : hostID{other.hostID},
        deviceID{other.deviceID},
        isMapped{other.isMapped}
  {
  }
};

template <size_t Teams, typename T, typename Reducer>
struct Reduce_Data {
  Offload_Info<Reducer>* info;
  T value;
  T *device;
  T *host;

  Reduce_Data() = delete;

  explicit Reduce_Data(T defaultValue, Offload_Info<Reducer> & oinfo)
      : info(&oinfo),
        value{defaultValue},
        device{(T *)omp_target_alloc(Teams * sizeof(T), info->deviceID)},
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
    std::fill_n(host, Teams, value);
    hostToDevice();
  }

  Reduce_Data(const Reduce_Data & other)
      : info{other.info}, value{other.value}, device{nullptr}, host{nullptr}
  {
  }

  void hostToDevice()
  {
    if (!!host && !!device
        && !omp_target_memcpy((void *)device,
                              (void *)host,
                              Teams * sizeof(T),
                              0,
                              0,
                              info->deviceID,
                              info->hostID)) {
      printf("Unable to copy memory from host to device\n");
      exit(1);
    }
  }

  void deviceToHost()
  {
    if (!!host && !!device
        && !omp_target_memcpy((void *)host,
                              (void *)device,
                              Teams * sizeof(T),
                              0,
                              0,
                              info->deviceID,
                              info->hostID)) {
      printf("Unable to copy memory from device to host\n");
      exit(1);
    }
  }

  void cleanup()
  {
    if (device) {
      omp_target_free((void *)device, info->deviceID);
      device = nullptr;
    }
    if (host) {
      delete[] host;
      host = nullptr;
    }
  }
};

#pragma omp declare target

template <typename T>
struct sum {
  static void apply(T &val, const T &v) { val += v; }
};

template <typename T>
struct min {
  static void apply(T &val, const T &v)
  {
    if (v < val) val = v;
  }
};

template <typename T>
struct max {
  static void apply(T &val, const T &v)
  {
    if (v > val) val = v;
  }
};

template <typename T, typename I = Index_type>
struct minloc {
  static void apply(I &loc, T &val, const I &l, const T &v)
  {
    if (v < val) {
      loc = l;
      val = v;
    }
  }
};

template <typename T, typename I = Index_type>
struct maxloc {
  static void apply(I &loc, T &val, const I &l, const T &v)
  {
    if (v > val) {
      loc = l;
      val = v;
    }
  }
};

#pragma omp end declare target

}  // end namespace omp


template <size_t Teams, typename T, typename Reducer>
struct TargetReduce
    : protected omp::Offload_Info<TargetReduce<Teams, T, Reducer>> {
  using SelfType = TargetReduce<Teams, T, Reducer>;
  using Offload = omp::Offload_Info<TargetReduce<Teams, T, Reducer>>;

  TargetReduce() = delete;
  TargetReduce(const TargetReduce &) = default;
  explicit TargetReduce(T init_val)
      : omp::Offload_Info<SelfType>(), val(init_val, *this)
  {
  }

  ~TargetReduce()
  {
    if (!omp_is_initial_device()) {
#pragma omp critical
      {
        int tid = omp_get_team_num();
        Reducer::apply(val.device[tid], val.value);
      }
    }
  }

  operator T()
  {
    if (!this->isMapped) {
      val.deviceToHost();
      for (int i = 0; i < Teams; ++i) {
        Reducer::apply(val.value, val.host[i]);
      }
      val.cleanup();
      this->isMapped = true;
    }
    return val.value;
  }
  T get() { return operator T(); }

  TargetReduce &reduce(T rhsVal)
  {
    Reducer::apply(val.value, rhsVal);
    return *this;
  }
  const TargetReduce &reduce(T rhsVal) const
  {
    return const_cast<SelfType *>(this)->reduce(rhsVal);
  }

private:
  omp::Reduce_Data<Teams, T, SelfType> val;
};

template <size_t Teams, typename T, typename Reducer>
struct TargetReduceLoc
    : protected omp::Offload_Info<TargetReduceLoc<Teams, T, Reducer>> {
  using SelfType = TargetReduceLoc<Teams, T, Reducer>;

  TargetReduceLoc() = delete;
  TargetReduceLoc(const TargetReduceLoc &) = default;
  explicit TargetReduceLoc(T init_val, Index_type init_loc)
      : omp::Offload_Info<SelfType>(),
        val(init_val, this->deviceID),
        loc(init_loc, this->deviceID)
  {
  }

  ~TargetReduceLoc()
  {
    if (!omp_is_initial_device()) {
#pragma omp critical
      {
        int tid = omp_get_team_num();
        Reducer::apply(loc.device[tid], val.host[tid], loc.value, val.value);
      }
    }
  }

  operator T()
  {
    if (!this->isMapped) {
      val.deviceToHost();
      loc.deviceToHost();
      for (int i = 0; i < Teams; ++i) {
        Reducer::apply(loc.value, val.value, loc.host[i], val.host[i]);
      }
      val.cleanup();
      loc.cleanup();
      this->isMapped = true;
    }
    return val.value;
  }
  T get() { return operator T(); }

  Index_type getLoc()
  {
    if (!this->isMapped) get();
    return loc.value;
  }

  TargetReduceLoc &reduce(T rhsVal, Index_type rhsLoc)
  {
    Reducer::apply(loc.value, val.value, rhsLoc, rhsVal);
    return *this;
  }
  const TargetReduceLoc &reduce(T rhsVal, Index_type rhsLoc) const
  {
    return const_cast<SelfType *>(this)->reduce(rhsVal, rhsLoc);
  }

private:
  omp::Reduce_Data<Teams, T, SelfType> val;
  omp::Reduce_Data<Teams, Index_type, SelfType> loc;
};

template <size_t Teams, typename T>
struct ReduceSum<omp_target_reduce<Teams>, T>
    : public TargetReduce<Teams, T, omp::sum<T>> {
  using parent = TargetReduce<Teams, T, omp::sum<T>>;
  using parent::parent;
  using parent::reduce;
  parent &operator+=(T rhsVal) { return reduce(rhsVal); }
  const parent &operator+=(T rhsVal) const { return reduce(rhsVal); }
};

template <size_t Teams, typename T>
struct ReduceMin<omp_target_reduce<Teams>, T>
    : public TargetReduce<Teams, T, omp::min<T>> {
  using parent = TargetReduce<Teams, T, omp::min<T>>;
  using parent::parent;
  using parent::reduce;
  parent &min(T rhsVal) { return reduce(rhsVal); }
  const parent &min(T rhsVal) const { return reduce(rhsVal); }
};

template <size_t Teams, typename T>
struct ReduceMax<omp_target_reduce<Teams>, T>
    : public TargetReduce<Teams, T, omp::max<T>> {
  using parent = TargetReduce<Teams, T, omp::max<T>>;
  using parent::parent;
  using parent::reduce;
  parent &max(T rhsVal) { return reduce(rhsVal); }
  const parent &max(T rhsVal) const { return reduce(rhsVal); }
};

template <size_t Teams, typename T>
struct ReduceMinLoc<omp_target_reduce<Teams>, T>
    : public TargetReduceLoc<Teams, T, omp::minloc<T>> {
  using parent = TargetReduceLoc<Teams, T, omp::minloc<T>>;
  using parent::parent;
  using parent::reduce;
  parent &minloc(Index_type rhsLoc, T rhsVal) { return reduce(rhsLoc, rhsVal); }
  const parent &minloc(Index_type rhsLoc, T rhsVal) const
  {
    return reduce(rhsLoc, rhsVal);
  }
};

template <size_t Teams, typename T>
struct ReduceMaxLoc<omp_target_reduce<Teams>, T>
    : public TargetReduceLoc<Teams, T, omp::maxloc<T>> {
  using parent = TargetReduceLoc<Teams, T, omp::maxloc<T>>;
  using parent::parent;
  using parent::reduce;
  parent &maxloc(Index_type rhsLoc, T rhsVal) { return reduce(rhsLoc, rhsVal); }
  const parent &maxloc(Index_type rhsLoc, T rhsVal) const
  {
    return reduce(rhsLoc, rhsVal);
  }
};

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_TARGET_OPENMP guard

#endif  // closing endif for header file include guard
