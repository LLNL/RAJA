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

template <size_t Teams, typename T>
struct Reduce_Data {
  mutable T value;
  T *device;
  T *host;

  Reduce_Data() = delete;

  explicit Reduce_Data(T defaultValue, Offload_Info &info)
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
    std::fill_n(host, Teams, value);
    hostToDevice(info);
  }

  Reduce_Data(const Reduce_Data &) = default;

  RAJA_INLINE void hostToDevice(Offload_Info &info)
  {
    // precondition: host and device are valid pointers
    if (omp_target_memcpy(reinterpret_cast<void *>(device),
                           reinterpret_cast<void *>(host),
                           Teams * sizeof(T),
                           0,
                           0,
                           info.deviceID,
                           info.hostID) != 0) {
      printf("Unable to copy memory from host to device\n");
      exit(1);
    }
  }
 
  RAJA_INLINE void deviceToHost(Offload_Info &info)
  {
    // precondition: host and device are valid pointers
    if (omp_target_memcpy(reinterpret_cast<void *>(host),
                          reinterpret_cast<void *>(device),
                           Teams * sizeof(T),
                           0,
                           0,
                           info.hostID,
                           info.deviceID) != 0) {
      printf("Unable to copy memory from device to host\n");
     exit(1);
    }
  }

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

#pragma omp declare target

template <typename T>
struct sum {
  void operator()(T &val, T v) { val += v; }
};

template <typename T>
struct min {
  void operator()(T &val, T v)
  {
    if (v < val) val = v;
  }
};

template <typename T>
struct max {
  void operator()(T &val, T v)
  {
    if (v > val) val = v;
  }
};

template <typename T, typename I>
struct minloc {
  void operator()(T &val, I &loc, T v, I l)
  {
    if (v < val) {
      loc = l;
      val = v;
    }
  }
};

template <typename T, typename I>
struct maxloc {
  void operator()(T &val, I &loc, T v, I l)
  {
    if (v > val) {
      loc = l;
      val = v;
    }
  }
};

#pragma omp end declare target

}  // end namespace omp


template <size_t Teams, typename Reducer, typename T>
struct TargetReduce {
  TargetReduce() = delete;
  TargetReduce(const TargetReduce &) = default; 

  explicit TargetReduce(T init_val) : info(), val(init_val, info)
  {
  }

  ~TargetReduce()
  {
    if (!omp_is_initial_device()) {
#pragma omp critical
      {
        int tid = omp_get_team_num();
        Reducer{}(val.device[tid], val.value);
      }
    }
  }

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
  T get() { return operator T(); }

  TargetReduce &reduce(T rhsVal)
  {
    Reducer{}(val.value, rhsVal);
    return *this;
  }
  const TargetReduce &reduce(T rhsVal) const
  {
    using NonConst = typename std::remove_const<decltype(this)>::type;
    auto ptr = const_cast<NonConst>(this); 
    Reducer{}(ptr->val.value,rhsVal);
    return *this;
  }

private:
  TargetReduce *ptr2this;
  omp::Offload_Info info;
  omp::Reduce_Data<Teams, T> val;
};

template <size_t Teams, typename Reducer, typename T, typename IndexType>
struct TargetReduceLoc {
  TargetReduceLoc() = delete;
  TargetReduceLoc(const TargetReduceLoc &) = default;
  explicit TargetReduceLoc(T init_val, IndexType init_loc)
      : info(), val(init_val, info), loc(init_loc, info)
  {
  }

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
  T get() { return operator T(); }

  IndexType getLoc()
  {
    if (!info.isMapped) get();
    return loc.value;
  }

  TargetReduceLoc &reduce(T rhsVal, IndexType rhsLoc)
  {
    Reducer{}(val.value, loc.value, rhsVal, rhsLoc);
    return *this;
  }
  const TargetReduceLoc &reduce(T rhsVal, IndexType rhsLoc) const
  {
    using NonConst = typename std::remove_const<decltype(this)>::type;
    auto ptr = const_cast<NonConst>(this);
    Reducer{}(ptr->val.value,ptr->loc.value,rhsVal,rhsLoc);
    return *this;

  }

private:
  omp::Offload_Info info;
  omp::Reduce_Data<Teams, T> val;
  omp::Reduce_Data<Teams, IndexType> loc;
};

template <size_t Teams, typename T>
struct ReduceSum<omp_target_reduce<Teams>, T>
    : public TargetReduce<Teams, omp::sum<T>, T> {
  using parent = TargetReduce<Teams, omp::sum<T>, T>;
  using parent::parent;
  parent &operator+=(T rhsVal) { return parent::reduce(rhsVal); }
  const parent &operator+=(T rhsVal) const { return parent::reduce(rhsVal); }
};

template <size_t Teams, typename T>
struct ReduceMin<omp_target_reduce<Teams>, T>
    : public TargetReduce<Teams, omp::min<T>, T> {
  using parent = TargetReduce<Teams, omp::min<T>, T>;
  using parent::parent;
  parent &min(T rhsVal) { return parent::reduce(rhsVal); }
  const parent &min(T rhsVal) const { return parent::reduce(rhsVal); }
};

template <size_t Teams, typename T>
struct ReduceMax<omp_target_reduce<Teams>, T>
    : public TargetReduce<Teams, omp::max<T>, T> {
  using parent = TargetReduce<Teams, omp::max<T>, T>;
  using parent::parent;
  parent &max(T rhsVal) { return parent::reduce(rhsVal); }
  const parent &max(T rhsVal) const { return parent::reduce(rhsVal); }
};

template <size_t Teams, typename T>
struct ReduceMinLoc<omp_target_reduce<Teams>, T>
    : public TargetReduceLoc<Teams, omp::minloc<T, Index_type>, T, Index_type> {
  using parent =
      TargetReduceLoc<Teams, omp::minloc<T, Index_type>, T, Index_type>;
  using parent::parent;
  parent &minloc(T rhsVal, Index_type rhsLoc)
  {
    return parent::reduce(rhsVal, rhsLoc);
  }
  const parent &minloc(T rhsVal, Index_type rhsLoc) const
  {
    return parent::reduce(rhsVal, rhsLoc);
  }
};

template <size_t Teams, typename T>
struct ReduceMaxLoc<omp_target_reduce<Teams>, T>
    : public TargetReduceLoc<Teams, omp::maxloc<T, Index_type>, T, Index_type> {
  using parent =
      TargetReduceLoc<Teams, omp::maxloc<T, Index_type>, T, Index_type>;
  using parent::parent;
  parent &maxloc(T rhsVal, Index_type rhsLoc)
  {
    return parent::reduce(rhsVal, rhsLoc);
  }
  const parent &maxloc(T rhsVal, Index_type rhsLoc) const
  {
    return parent::reduce(rhsVal, rhsLoc);
  }
};


/* IGNORE FOR NOW -- FUTURE MAGIC

#define DO_IN_ORDER(UnexpandedPack)                     \
  do {                                                  \
    int unused[] = {0, ((void)(UnexpandedPack), 0)...}; \
    (void)unused;                                       \
  } while (0)

namespace detail
{
template <size_t, typename, typename, typename...>
struct TargetReduceGeneric;

template <size_t Teams, template <typename...> class Reducer, typename... Ts, size_t... Indicies>
struct TargetReduceGeneric<Teams,
                           Reducer<Ts...>,
                           VarOps::index_sequence<Indicies...>,
                           Ts...> {
  using SelfType = TargetReduceGeneric<Teams, Reducer<Ts...>, VarOps::index_sequence<Indicies...>, Ts...>;

  TargetReduceLoc() = delete;
  TargetReduceLoc(const TargetReduceGeneric &) = default;
  explicit TargetReduceGeneric(Ts... init)
      : info(), data(omp::Reduce_Data<Teams, Ts>(init, info)...)
  {
  }

  ~TargetReduceGeneric()
  {
    if (!omp_is_initial_device()) {
#pragma omp critical
      {
        int tid = omp_get_team_num();
        Reducer{}(std::get<Indicies>(data).device...,
                       std::get<Indicies>(data).val...);
      }
    }
  }

  void mapToHost
  {
    if (!info.isMapped) {
      DO_IN_ORDER(std::get<Indicies>(data).deviceToHost(info));
      for (int i = 0; i < Teams; ++i) {
        Reducer{}(std::get<Indicies>(data).val...,
                       std::get<Indicies>(data).host);
      }
      DO_IN_ORDER(std::get<Indicies>(data).cleanup(info));
      info.isMapped = true;
    }
    return std::get<0>(data).value;
  }

  template <unsigned int I>
  auto get() -> decltype(std::get<I>(data).value)
  {
    if (!info.isMapped) mapToHost();
    return std::get<I>(data).value;
  }

  TargetReduceLoc &reduce(T rhsVal, IndexType rhsLoc)
  {
    Reducer{}(std::get<Indicies>(data).value..., args...);
    return *this;
  }
  const TargetReduceLoc &reduce(T rhsVal, IndexType rhsLoc) const
  {
    return const_cast<SelfType *>(this)->reduce(rhsVal, rhsLoc);
  }

private:
  omp::Offload_Info info;
  std::tuple<Ts...> data;
};
}  // end namespace detail

#undef DO_IN_ORDER

template <size_t Teams, template <typename...> class Reducer, typename... Types>
struct TargetReduceGeneric
    : public detail::
          TargetReduceGeneric<Teams,
                              Reducer,
                              VarOps::make_index_sequence<sizeof...(Types)>,
                              Types...> {
};

template <size_t Teams, template <typename...> class Reducer, typename ... Ts>
struct VariadicTargetReduce : public TargetReduceGeneric<Teams, Reducer<Ts...>, Ts...> {
  using parent = TargetReduceGeneric<Teams, Reducer<Ts...>, Ts...>
  using parent::parent;
};

template <size_t N, typename T>
struct ReduceSum<omp_target_reduce<N>, T> : public VariadicTargetReduce<N, omp::sum, T> {
  using parent = VariadicTargetReduce<N, omp::sum, T>;
  using parent::parent;
  parent &operator+=(T rhsVal) { return parent::reduce(rhsVal); }
  const parent &operator+=(T rhsVal) const { return parent::reduce(rhsVal); }
};
template <size_t N, typename T>
struct ReduceMin<omp_target_reduce<N>, T> : public VariadicTargetReduce<N, omp::min, T> {
  using parent = VariadicTargetReduce<N, omp::min, T>;
  using parent::parent;
  parent &min(T rhsVal) { return parent::reduce(rhsVal); }
  const parent &min(T rhsVal) const { return parent::reduce(rhsVal); }
};
template <size_t N, typename T>
struct ReduceMax<omp_target_reduce<N>, T> : public VariadicTargetReduce<N, omp::max, T> {
  using parent = VariadicTargetReduce<N, omp::max, T>;
  using parent::parent;
  parent &max(T rhsVal) { return parent::reduce(rhsVal); }
  const parent &max(T rhsVal) const { return parent::reduce(rhsVal); }
};
template <size_t N, typename T>
struct ReduceMinLoc<omp_target_reduce<N>, T> : public VariadicTargetReduce<N, omp::minloc, T, Index_type> {
  using parent = VariadicTargetReduce<N, omp::minloc, T, Index_type>;
  using parent::parent;
  parent &min(T rhsVal, Index_type rhsLoc) { return parent::reduce(rhsVal, rhsLoc); }
  const parent &min(T rhsVal, Index_type rhsLoc) const { return parent::reduce(rhsVal, rhsLoc); }
};
template <size_t N, typename T>
struct ReduceMaxLoc<omp_target_reduce<N>, T> : public VariadicTargetReduce<N, omp::maxloc, T, Index_type> {
  using parent = VariadicTargetReduce<N, omp::max, T>;
  using parent::parent;
  parent &max(T rhsVal, Index_type rhsLoc) { return parent::reduce(rhsVal, rhsLoc); }
  const parent &max(T rhsVal, Index_type rhsLoc) const { return parent::reduce(rhsVal, rhsLoc); }
};

*/

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_TARGET_OPENMP guard

#endif  // closing endif for header file include guard
