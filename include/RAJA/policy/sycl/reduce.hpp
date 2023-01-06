/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for SYCL reduction stucts/classes.
 *          
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_sycl_reduce_HPP
#define RAJA_sycl_reduce_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <algorithm>


#include "RAJA/util/types.hpp"

#include "RAJA/pattern/reduce.hpp"

#include "RAJA/policy/sycl/policy.hpp"

namespace RAJA
{

namespace sycl
{

template <typename T, typename I>
struct minloc 
{
  static constexpr T identity = T(::RAJA::operators::limits<T>::max());
  RAJA_HOST_DEVICE RAJA_INLINE void operator()(T &val,
                                               I &loc,
                                               const T v,
                                               const I l)
  {
    if (v < val) {
      loc = l;
      val = v;
    }
  }
};

template <typename T, typename I>
struct maxloc 
{
  static constexpr T identity = T(::RAJA::operators::limits<T>::min());
  RAJA_HOST_DEVICE RAJA_INLINE void operator()(T &val,
                                               I &loc,
                                               const T v,
                                               const I l)
  {
    if (v > val) {
      loc = l;
      val = v;
    }
  }
};

// Alias for clarity. Reduction size operates on number of  teams.
// Ideally, MaxNumTeams = ThreadsPerTeam in omp_target_parallel_for_exec.
static int MaxNumTeams = 1;

//! Information necessary for OpenMP offload to be considered
struct Offload_Info 
{
  int hostID{1};
  int deviceID{2};
  bool isMapped{false};

  Offload_Info() = default;

  Offload_Info(const Offload_Info &other)
      : hostID{other.hostID}, deviceID{other.deviceID}, isMapped{other.isMapped}
  {
  }
};

//! Reduction data for OpenMP Offload -- stores value, host pointer, and device
//! pointer
template <typename T>
struct Reduce_Data
{
  mutable T value;
  T *device;
  T *host;

  //! disallow default constructor
  Reduce_Data() = delete;

  /*! \brief create from a default value and offload information
   *
   *  allocates data on the host and device and initializes values to default
   */
  Reduce_Data(T initValue, T identityValue, Offload_Info &info)
      : value(initValue)
  {
    cl::sycl::queue* q = ::RAJA::sycl::detail::getQueue();

    if(!q) {
      camp::resources::Resource res = camp::resources::Sycl();
      q = res.get<camp::resources::Sycl>().get_queue();
    } 

    device = reinterpret_cast<T *>(cl::sycl::malloc_device(sycl::MaxNumTeams * sizeof(T), *(q)));
    host = reinterpret_cast<T *>(cl::sycl::malloc_host(sycl::MaxNumTeams * sizeof(T), *(q)));

    if (!host) {
      printf("Unable to allocate space on host\n");
      exit(1);
    }
    if (!device) {
      printf("Unable to allocate space on device\n");
      exit(1);
    }
    std::fill_n(host, sycl::MaxNumTeams, identityValue);
    hostToDevice(info);
  }

  void reset(T initValue)
  {
    value = initValue;
  }

  //! default copy constructor for POD
  Reduce_Data(const Reduce_Data &) = default;

  //! transfers from the host to the device -- exit() is called upon failure
  RAJA_INLINE void hostToDevice(Offload_Info &info)
  {
    cl::sycl::queue* q = ::RAJA::sycl::detail::getQueue();

    if(!q) {
      camp::resources::Resource res = camp::resources::Sycl();
      q = res.get<camp::resources::Sycl>().get_queue();
    }

    // precondition: host and device are valid pointers
    auto e = q->memcpy(reinterpret_cast<void *>(device),
                       reinterpret_cast<void *>(host),
                       sycl::MaxNumTeams * sizeof(T));

    e.wait();
  }

  //! transfers from the device to the host -- exit() is called upon failure
  RAJA_INLINE void deviceToHost(Offload_Info &info)
  {
    cl::sycl::queue* q = ::RAJA::sycl::detail::getQueue();

    if(!q) {
      camp::resources::Resource res = camp::resources::Sycl();
      q = res.get<camp::resources::Sycl>().get_queue();
    } 

    // precondition: host and device are valid pointers
    auto e = q->memcpy(reinterpret_cast<void *>(host),
                       reinterpret_cast<void *>(device),
                       sycl::MaxNumTeams * sizeof(T));
    
    e.wait();
  }

  //! frees all data from the offload information passed
  RAJA_INLINE void cleanup(Offload_Info &info)
  {
    cl::sycl::queue* q = ::RAJA::sycl::detail::getQueue();
    if(!q) {
      camp::resources::Resource res = camp::resources::Sycl();
      q = res.get<camp::resources::Sycl>().get_queue();
    }
    if (device) {
      cl::sycl::free(reinterpret_cast<void *>(device), *q);
      device = nullptr;
    }
    if (host) {
      cl::sycl::free(reinterpret_cast<void *>(host), *q);
      //delete[] host;
      host = nullptr;
    }
  }
};

}  // end namespace sycl

//! OpenMP Target Reduction entity -- generalize on # of teams, reduction, and
//! type
template <typename Reducer, typename T>
struct TargetReduce 
{
  TargetReduce() = delete;
  TargetReduce(const TargetReduce &) = default;

  explicit TargetReduce(T init_val)
      : info(),
        val(Reducer::identity(), Reducer::identity(), info),
        initVal(init_val),
        finalVal(Reducer::identity())
  {
  }

  void reset(T init_val_, T identity_ = Reducer::identity())
  {
    val.cleanup(info);
    val = sycl::Reduce_Data<T>(identity_, identity_, info);
    info.isMapped = false;
    initVal = init_val_;
    finalVal = identity_;
  }

  //! apply reduction on device upon destruction
  ~TargetReduce()
  {
  }

  //! map result value back to host if not done already; return aggregate value
  operator T()
  {
    if (!info.isMapped) {
      val.deviceToHost(info);
      for (int i =0; i < sycl::MaxNumTeams; ++i) {
        Reducer{}(val.value, val.host[i]);
      }
//      val.cleanup(info);
      info.isMapped = true;
    }
    finalVal = Reducer::identity();
    Reducer{}(finalVal, initVal);
    Reducer{}(finalVal, val.value);
    T returnVal = finalVal;
    reset(finalVal);
    return returnVal;
  }
  //! alias for operator T()
  T get() { return operator T(); }

  //! apply reduction
  TargetReduce &reduce(T rhsVal)
  {
#ifdef __SYCL_DEVICE_ONLY__
    auto i = 0; //__spirv::initLocalInvocationId<1, cl::sycl::id<1>>()[0];
    auto atm = cl::sycl::ext::oneapi::atomic_ref<T, cl::sycl::memory_order_acq_rel, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space>(val.device[i]);
    Reducer{}(atm, rhsVal);
    return *this;
#else
    Reducer{}(val.value, rhsVal);
    return *this;
#endif
  }

  //! apply reduction (const version) -- still reduces internal values
  const TargetReduce &reduce(T rhsVal) const
  {
#ifdef __SYCL_DEVICE_ONLY__
    auto i = 0; //__spirv::initLocalInvocationId<1, cl::sycl::id<1>>()[0];
    auto atm = cl::sycl::ext::oneapi::atomic_ref<T, cl::sycl::memory_order_acq_rel, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space>(val.device[i]);
    Reducer{}(atm, rhsVal);  
    return *this;
#else
    Reducer{}(val.value, rhsVal);
    return *this;
#endif
  }

  //! storage for reduction data (host ptr, device ptr, value)
  sycl::Reduce_Data<T> val;

private:
  //! storage for offload information (host ID, device ID)
  sycl::Offload_Info info;
  //! storage for reduction data (host ptr, device ptr, value)
  T initVal;
  T finalVal;
};

//! OpenMP Target Reduction Location entity -- generalize on # of teams,
//! reduction, and type
template <typename Reducer, typename T, typename IndexType>
struct TargetReduceLoc 
{
  TargetReduceLoc() = delete;
  TargetReduceLoc(const TargetReduceLoc &) = default;
  explicit TargetReduceLoc(T init_val, IndexType init_loc)
      : info(),
        val(Reducer::identity, Reducer::identity, info),
        loc(init_loc, IndexType(RAJA::reduce::detail::DefaultLoc<IndexType>().value()), info),
        initVal(init_val),
        finalVal(Reducer::identity),
        initLoc(init_loc),
        finalLoc(IndexType(RAJA::reduce::detail::DefaultLoc<IndexType>().value()))
  {
  }

  //! apply reduction on device upon destruction
  ~TargetReduceLoc()
  {
  }

  //! map result value back to host if not done already; return aggregate value
  operator T()
  {
    if (!info.isMapped) {
      val.deviceToHost(info);
      loc.deviceToHost(info);
      
      for (int i = 0; i < sycl::MaxNumTeams; ++i) {
        Reducer{}(val.value, loc.value, val.host[i], loc.host[i]);
      }
      info.isMapped = true;
    }
    finalVal = Reducer::identity;
    finalLoc = IndexType(RAJA::reduce::detail::DefaultLoc<IndexType>().value());
    Reducer{}(finalVal, finalLoc, initVal, initLoc);
    Reducer{}(finalVal, finalLoc, val.value, loc.value);
    returnVal = finalVal;
    returnLoc = finalLoc;
    reset(finalVal, finalLoc);
    return returnVal;
  }
  //! alias for operator T()
  T get() { return operator T(); }

  void reset(T init_val_,
             IndexType init_local_ =
             IndexType(RAJA::reduce::detail::DefaultLoc<IndexType>().value()),
             T identity_ = Reducer::identity)
  {
    val.cleanup(info);
    val = sycl::Reduce_Data<T>(identity_, identity_, info);
    loc.cleanup(info);
    loc = sycl::Reduce_Data<IndexType>(reduce::detail::DefaultLoc<IndexType>().value(), reduce::detail::DefaultLoc<IndexType>().value(), info);
    info.isMapped = false;
    initVal = init_val_;
    finalVal = identity_;
    initLoc = init_local_;//IndexType(RAJA::reduce::detail::DefaultLoc<IndexType>().value());
    finalLoc = IndexType(RAJA::reduce::detail::DefaultLoc<IndexType>().value());
  }


  //! map result value back to host if not done already; return aggregate
  //! location
  IndexType getLoc()
  {
    if (!info.isMapped) get();
    // return loc.value;
    return (returnLoc);
  }

  //! apply reduction
  TargetReduceLoc &reduce(T rhsVal, IndexType rhsLoc)
  {
#ifdef __SYCL_DEVICE_ONLY__
    auto i = 0; //__spirv::initLocalInvocationId<1, cl::sycl::id<1>>()[0];
    cl::sycl::atomic_fence(cl::sycl::memory_order_acquire, cl::sycl::memory_scope::device);
    Reducer{}(val.device[i], loc.device[i], rhsVal, rhsLoc);
    cl::sycl::atomic_fence(cl::sycl::memory_order_release, cl::sycl::memory_scope::device);
    return *this;
#else
    Reducer{}(val.value, loc.value, rhsVal, rhsLoc);
    return *this;
#endif
  }

  //! apply reduction (const version) -- still reduces internal values
  const TargetReduceLoc &reduce(T rhsVal, IndexType rhsLoc) const
  {
    Reducer{}(val.value, loc.value, rhsVal, rhsLoc);
    return *this;
  }

  //! storage for reduction data for value
  sycl::Reduce_Data<T> val;
  sycl::Reduce_Data<IndexType> loc;

private:
  //! storage for offload information
  sycl::Offload_Info info;
  //! storage for reduction data for value
//  sycl::Reduce_Data<T> val;
  //! storage for redcution data for location
  T initVal;
  T finalVal;
  T returnVal;
  IndexType initLoc;
  IndexType finalLoc;
  IndexType returnLoc;
};


//! specialization of ReduceSum for omp_target_reduce
template <typename T>
class ReduceSum<sycl_reduce, T>
    : public TargetReduce<RAJA::reduce::sum<T>, T>
{
public:

  using self = ReduceSum<sycl_reduce, T>;
  using parent = TargetReduce<RAJA::reduce::sum<T>, T>;
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
#ifdef __SYCL_DEVICE_ONLY__
    auto i = 0;//__spirv::initLocalInvocationId<1, cl::sycl::id<1>>()[0];
    auto atm = cl::sycl::ext::oneapi::atomic_ref<T, cl::sycl::memory_order_acq_rel, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space>(parent::val.device[i]);
    atm.fetch_add(rhsVal);
    return *this;
#else
    parent::reduce(rhsVal);
    return *this;
#endif
  }
};

//! specialization of ReduceBitOr for sycl_reduce
template <typename T>
class ReduceBitOr<sycl_reduce, T>
    : public TargetReduce<RAJA::reduce::or_bit<T>, T>
{
public:

  using self = ReduceBitOr<sycl_reduce, T>;
  using parent = TargetReduce<RAJA::reduce::or_bit<T>, T>;
  using parent::parent;

  //! enable operator|= for ReduceBitOr -- alias for reduce()
  self &operator|=(T rhsVal)
  {
#ifdef __SYCL_DEVICE_ONLY__
    auto i = 0;//__spirv::initLocalInvocationId<1, cl::sycl::id<1>>()[0];
    auto atm = cl::sycl::ext::oneapi::atomic_ref<T, cl::sycl::memory_order_acq_rel, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space>(parent::val.device[i]);
    atm |= rhsVal;
    return *this;
#else
    parent::reduce(rhsVal);
    return *this;
#endif
  }

  //! enable operator|= for ReduceBitOr -- alias for reduce()
  const self &operator|=(T rhsVal) const
  {
#ifdef __SYCL_DEVICE_ONLY__
    auto i = 0;//__spirv::initLocalInvocationId<1, cl::sycl::id<1>>()[0];
    auto atm = cl::sycl::ext::oneapi::atomic_ref<T, cl::sycl::memory_order_acq_rel, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space>(parent::val.device[i]);
    atm |= rhsVal;
    return *this;
#else
    parent::reduce(rhsVal);
    return *this;
#endif
  }
};

//! specialization of ReduceBitAnd for sycl_reduce
template <typename T>
class ReduceBitAnd<sycl_reduce, T>
    : public TargetReduce<RAJA::reduce::and_bit<T>, T>
{
public:

  using self = ReduceBitAnd<sycl_reduce, T>;
  using parent = TargetReduce<RAJA::reduce::and_bit<T>, T>;
  using parent::parent;

  //! enable operator&= for ReduceBitAnd -- alias for reduce()
  self &operator&=(T rhsVal)
  {
#ifdef __SYCL_DEVICE_ONLY__
    auto i = 0;//__spirv::initLocalInvocationId<1, cl::sycl::id<1>>()[0];
    auto atm = cl::sycl::ext::oneapi::atomic_ref<T, cl::sycl::memory_order_acq_rel, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space>(parent::val.device[i]);
    atm &= rhsVal;
    return *this;
#else
    parent::reduce(rhsVal);
    return *this;
#endif
  }

  //! enable operator&= for ReduceBitAnd -- alias for reduce()
  const self &operator&=(T rhsVal) const
  {
#ifdef __SYCL_DEVICE_ONLY__
    auto i = 0;//__spirv::initLocalInvocationId<1, cl::sycl::id<1>>()[0];
    auto atm = cl::sycl::ext::oneapi::atomic_ref<T, cl::sycl::memory_order_acq_rel, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space>(parent::val.device[i]);
    atm &= rhsVal;
    return *this;
#else
    parent::reduce(rhsVal);
    return *this;
#endif
  }
};


//! specialization of ReduceMin for omp_target_reduce
template <typename T>
class ReduceMin<sycl_reduce, T>
    : public TargetReduce<RAJA::reduce::min<T>, T>
{
public:

  using self = ReduceMin<sycl_reduce, T>;
  using parent = TargetReduce<RAJA::reduce::min<T>, T>;
  using parent::parent;

  //! enable min() for ReduceMin -- alias for reduce()
  self &min(T rhsVal)
  {
#ifdef __SYCL_DEVICE_ONLY__
    auto i = 0;//__spirv::initLocalInvocationId<1, cl::sycl::id<1>>()[0];
    auto atm = cl::sycl::ext::oneapi::atomic_ref<T, cl::sycl::memory_order_acq_rel, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space>(parent::val.device[i]);
    atm.fetch_min(rhsVal);
    return *this;
#else
    parent::reduce(rhsVal);
    return *this;
#endif
  }

  //! enable min() for ReduceMin -- alias for reduce()
  const self &min(T rhsVal) const
  {
#ifdef __SYCL_DEVICE_ONLY__
    auto i = 0;//__spirv::initLocalInvocationId<1, cl::sycl::id<1>>()[0];
    auto atm = cl::sycl::ext::oneapi::atomic_ref<T, cl::sycl::memory_order_acq_rel, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space>(parent::val.device[i]);
    atm.fetch_min(rhsVal);
    return *this;
#else
    parent::reduce(rhsVal);
    return *this;
#endif
  }
};


//! specialization of ReduceMax for omp_target_reduce
template <typename T>
class ReduceMax<sycl_reduce, T>
    : public TargetReduce<RAJA::reduce::max<T>, T>
{
public:

  using self = ReduceMax<sycl_reduce, T>;
  using parent = TargetReduce<RAJA::reduce::max<T>, T>;
  using parent::parent;

  //! enable max() for ReduceMax -- alias for reduce()
  self &max(T rhsVal)
  {
#ifdef __SYCL_DEVICE_ONLY__
    auto i = 0;//__spirv::initLocalInvocationId<1, cl::sycl::id<1>>()[0];
    auto atm = cl::sycl::ext::oneapi::atomic_ref<T, cl::sycl::memory_order_acq_rel, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space>(parent::val.device[i]);
    atm.fetch_max(rhsVal);
    return *this;
#else
    parent::reduce(rhsVal);
    return *this;
#endif
  }

  //! enable max() for ReduceMax -- alias for reduce()
  const self &max(T rhsVal) const
  {
#ifdef __SYCL_DEVICE_ONLY__
    auto i = 0;//__spirv::initLocalInvocationId<1, cl::sycl::id<1>>()[0];
    auto atm = cl::sycl::ext::oneapi::atomic_ref<T, cl::sycl::memory_order_acq_rel, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space>(parent::val.device[i]);
    atm.fetch_max(rhsVal);
    return *this;
#else
    parent::reduce(rhsVal);
    return *this;
#endif
  }
};

//! specialization of ReduceMinLoc for omp_target_reduce
template <typename T, typename IndexType>
class ReduceMinLoc<sycl_reduce, T, IndexType>
    : public TargetReduceLoc<sycl::minloc<T, IndexType>, T, IndexType>
{
public:

  using self = ReduceMinLoc<sycl_reduce, T, IndexType>;
  using parent =
      TargetReduceLoc<sycl::minloc<T, IndexType>, T, IndexType>;
  using parent::parent;

  //! enable minloc() for ReduceMinLoc -- alias for reduce()
  self &minloc(T rhsVal, IndexType rhsLoc)
  {
#ifdef __SYCL_DEVICE_ONLY__
    // TODO: Race condition currently
    auto i = 0;//__spirv::initLocalInvocationId<1, cl::sycl::id<1>>()[0];
    auto atm = cl::sycl::ext::oneapi::atomic_ref<T, cl::sycl::memory_order_acq_rel, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space>(parent::val.device[i]);
    auto oldMin = atm.fetch_min(rhsVal);
    if(oldMin >= rhsVal) { // New min or Same min
      if(oldMin == rhsVal) { // Same as old min
        if(rhsLoc < parent::loc.device[i]) { // if same, only overwrite if earlier
          if(rhsVal == atm.load()) {
            parent::loc.device[i] = rhsLoc;
	  }
        }
      } else {
        if(rhsVal == atm.load()) {
          parent::loc.device[i] = rhsLoc;
	}
      }
    }
    return *this;
#else
    parent::reduce(rhsVal, rhsLoc);
    return *this;
#endif
  }

  //! enable minloc() for ReduceMinLoc -- alias for reduce()
  const self &minloc(T rhsVal, IndexType rhsLoc) const
  {
#ifdef __SYCL_DEVICE_ONLY__
    // TODO: Race condition currently
    auto i = 0;//__spirv::initLocalInvocationId<1, cl::sycl::id<1>>()[0];
    auto atm = cl::sycl::ext::oneapi::atomic_ref<T, cl::sycl::memory_order_acq_rel, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space>(parent::val.device[i]);
    auto oldMin = atm.fetch_min(rhsVal);
    if(oldMin >= rhsVal) { // New min or Same min
      if(oldMin == rhsVal) { // Same as old min
        if(rhsLoc < parent::loc.device[i]) { // if same, only overwrite if earlier
          if(rhsVal == atm.load()) {
            parent::loc.device[i] = rhsLoc;
	  }
        }
      } else {
        if(rhsVal == atm.load()) {
          parent::loc.device[i] = rhsLoc;
	}
      }
    }
    return *this;
#else
    parent::reduce(rhsVal, rhsLoc);
    return *this;
#endif
  }
};


//! specialization of ReduceMaxLoc for omp_target_reduce
template <typename T, typename IndexType>
class ReduceMaxLoc<sycl_reduce, T, IndexType>
    : public TargetReduceLoc<sycl::maxloc<T, IndexType>, T, IndexType>
{
public:

  using self = ReduceMaxLoc<sycl_reduce, T, IndexType>;
  using parent =
      TargetReduceLoc<sycl::maxloc<T, IndexType>, T, IndexType>;
  using parent::parent;

  //! enable maxloc() for ReduceMaxLoc -- alias for reduce()
  self &maxloc(T rhsVal, IndexType rhsLoc)
  {
#ifdef __SYCL_DEVICE_ONLY__
    // TODO: Race condition currently
    auto i = 0;//__spirv::initLocalInvocationId<1, cl::sycl::id<1>>()[0];
    auto atm = cl::sycl::ext::oneapi::atomic_ref<T, cl::sycl::memory_order_acq_rel, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space>(parent::val.device[i]);
    auto oldMin = atm.fetch_max(rhsVal);
    if(oldMin <= rhsVal) { // New min or Same min
      if(oldMin == rhsVal) { // Same as old min
        if(rhsLoc < parent::loc.device[i]) { // if same, only overwrite if earlier
          if(rhsVal == atm.load()) {
            parent::loc.device[i] = rhsLoc;
	  }
        }
      } else {
        if(rhsVal == atm.load()) {
          parent::loc.device[i] = rhsLoc;
	}
      }
    }
    return *this;
#else
    parent::reduce(rhsVal, rhsLoc);
    return *this;
#endif
  }

  //! enable maxloc() for ReduceMaxLoc -- alias for reduce()
  const self &maxloc(T rhsVal, IndexType rhsLoc) const
  {
#ifdef __SYCL_DEVICE_ONLY__
    // TODO: Race condition currently
    auto i = 0;//__spirv::initLocalInvocationId<1, cl::sycl::id<1>>()[0];
    auto atm = cl::sycl::ext::oneapi::atomic_ref<T, cl::sycl::memory_order_acq_rel, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space>(parent::val.device[i]);
    auto oldMin = atm.fetch_max(rhsVal);
    if(oldMin <= rhsVal) { // New min or Same min
      if(oldMin == rhsVal) { // Same as old min
        if(rhsLoc < parent::loc.device[i]) { // if same, only overwrite if earlier
          if(rhsVal == atm.load()) {
            parent::loc.device[i] = rhsLoc;
	  }
        }
      } else {
        if(rhsVal == atm.load()) {
          parent::loc.device[i] = rhsLoc;
	}
      }
    }
    return *this;
#else
    parent::reduce(rhsVal, rhsLoc);
    return *this;
#endif
  }
};


}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_SYCL guard

#endif  // closing endif for header file include guard
