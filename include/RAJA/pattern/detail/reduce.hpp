#ifndef RAJA_PATTERN_DETAIL_REDUCE_HPP
#define RAJA_PATTERN_DETAIL_REDUCE_HPP

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


#include "RAJA/util/Operators.hpp"
#include "RAJA/util/types.hpp"

#define RAJA_DECLARE_REDUCER(OP, POL, COMBINER)               \
  template <typename T>                                       \
  class Reduce##OP<POL, T>                                    \
      : public reduce::detail::BaseReduce##OP<T, COMBINER>    \
  {                                                           \
  public:                                                     \
    using Base = reduce::detail::BaseReduce##OP<T, COMBINER>; \
    using Base::Base;                                         \
  };

#define RAJA_DECLARE_ALL_REDUCERS(POL, COMBINER) \
  RAJA_DECLARE_REDUCER(Sum, POL, COMBINER)       \
  RAJA_DECLARE_REDUCER(Min, POL, COMBINER)       \
  RAJA_DECLARE_REDUCER(Max, POL, COMBINER)       \
  RAJA_DECLARE_REDUCER(MinLoc, POL, COMBINER)    \
  RAJA_DECLARE_REDUCER(MaxLoc, POL, COMBINER)

namespace RAJA
{

namespace reduce
{

#ifdef RAJA_ENABLE_TARGET_OPENMP
#pragma omp declare target
#endif

namespace detail
{

template <typename T, template <typename...> class Op>
struct op_adapter : private Op<T, T, T> {
  using operator_type = Op<T, T, T>;
  RAJA_HOST_DEVICE static constexpr T identity() {
    return operator_type::identity();
  }
  RAJA_HOST_DEVICE RAJA_INLINE void operator()(T &val, const T v) const
  {
    val = operator_type::operator()(val, v);
  }
};
}  // end detail

template <typename T>
struct sum : detail::op_adapter<T, RAJA::operators::plus> {
};

template <typename T>
struct min : detail::op_adapter<T, RAJA::operators::minimum> {
};

template <typename T>
struct max : detail::op_adapter<T, RAJA::operators::maximum> {
};

#ifdef RAJA_ENABLE_TARGET_OPENMP
#pragma omp end declare target
#endif

namespace detail
{

template <typename T, bool doing_min = true>
class ValueLoc
{
public:
  T val = doing_min ? operators::limits<T>::max() : operators::limits<T>::min();
  Index_type loc = -1;

  RAJA_HOST_DEVICE constexpr ValueLoc() = default;
  RAJA_HOST_DEVICE constexpr ValueLoc(ValueLoc const &) = default;
  RAJA_HOST_DEVICE ValueLoc &operator=(ValueLoc const &) = default;
  RAJA_HOST_DEVICE constexpr ValueLoc(T const &val) : val{val}, loc{-1} {}
  RAJA_HOST_DEVICE constexpr ValueLoc(T const &val, Index_type const &loc)
      : val{val}, loc{loc}
  {
  }

  RAJA_HOST_DEVICE operator T() const { return val; }
  RAJA_HOST_DEVICE Index_type getLoc() { return loc; }
  RAJA_HOST_DEVICE bool operator<(ValueLoc const &rhs) const
  {
    return val < rhs.val;
  }
  RAJA_HOST_DEVICE bool operator>(ValueLoc const &rhs) const
  {
    return val > rhs.val;
  }
};

}  // end detail

}  // end reduce

namespace operators
{
template <typename T, bool B>
struct limits<::RAJA::reduce::detail::ValueLoc<T, B>> : limits<T> {
};
}

namespace reduce
{

namespace detail
{

template <typename T,
          template <typename> class Reduce_,
          template <typename, typename> class Combiner_>
class BaseReduce
{
  using Reduce = Reduce_<T>;
  // NOTE: the _t here is to appease MSVC
  using Combiner_t = Combiner_<T, Reduce>;
  Combiner_t mutable c;

public:
  using value_type = T;
  using reduce_type = Reduce;

  //! prohibit compiler-generated default ctor
  BaseReduce() = delete;

  //! prohibit compiler-generated copy assignment
  BaseReduce &operator=(const BaseReduce &) = delete;

  //! compiler-generated copy constructor
  BaseReduce(const BaseReduce &) = default;

  //! compiler-generated move constructor
  BaseReduce(BaseReduce &&) = default;

  //! compiler-generated move assignment
  BaseReduce &operator=(BaseReduce &&) = default;

  constexpr BaseReduce(T init_val, T identity_ = Reduce::identity())
      : c{init_val, identity_}
  {
  }

  void combine(T const &other) const { c.combine(other); }

  T &local() const { return c.local(); }

  //! Get the calculated reduced value
  operator T() const { return c.get(); }

  //! Get the calculated reduced value
  T get() const { return c.get(); }
};

template <typename T, typename Reduce, typename Derived>
class BaseCombinable
{
protected:
  BaseCombinable const *parent = nullptr;
  T identity;
  T mutable my_data;

public:
  //! prohibit compiler-generated default ctor
  BaseCombinable() = delete;

  constexpr BaseCombinable(T init_val, T identity_ = T())
      : identity{identity_}, my_data{init_val}
  {
  }

  constexpr BaseCombinable(BaseCombinable const &other)
      : parent{other.parent ? other.parent : &other},
        identity{other.identity},
        my_data{identity}
  {
  }

  ~BaseCombinable()
  {
    if (parent && my_data != identity) {
      Reduce()(parent->my_data, my_data);
    }
  }

  void combine(T const &other) { Reduce{}(my_data, other); }

  /*!
   *  \return the calculated reduced value
   */
  T get() const { return derived().get_combined(); }

  /*!
   *  \return reference to the local value
   */
  T &local() const { return my_data; }

  T get_combined() const { return my_data; }

private:
  // Convenience method for CRTP
  const Derived &derived() const { return *(static_cast<const Derived *>(this)); }
  Derived &derived() { return *(static_cast<Derived *>(this)); }
};

/*!
 ******************************************************************************
 *
 * \brief  Min reducer class template for use in OpenMP execution.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename T, template <typename, typename> class Combiner>
class BaseReduceMin : public BaseReduce<T, RAJA::reduce::min, Combiner>
{
public:
  using Base = BaseReduce<T, RAJA::reduce::min, Combiner>;
  using Base::Base;

  //! reducer function; updates the current instance's state
  const BaseReduceMin &min(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }
};

/*!
 **************************************************************************
 *
 * \brief  MinLoc reducer class template for use in tbb execution.
 *
 **************************************************************************
 */
template <typename T, template <typename, typename> class Combiner>
class BaseReduceMinLoc
    : public BaseReduce<ValueLoc<T>, RAJA::reduce::min, Combiner>
{
public:
  using Base = BaseReduce<ValueLoc<T>, RAJA::reduce::min, Combiner>;
  using value_type = typename Base::value_type;
  using Base::Base;

  //! constructor requires a default value for the reducer
  explicit BaseReduceMinLoc(T init_val, Index_type init_idx)
      : Base(value_type(init_val, init_idx))
  {
  }

  /// \brief reducer function; updates the current instance's state
  const BaseReduceMinLoc &minloc(T rhs, Index_type loc) const
  {
    this->combine(value_type(rhs, loc));
    return *this;
  }

  //! Get the calculated reduced value
  Index_type getLoc() const { return Base::get().getLoc(); }

  //! Get the calculated reduced value
  operator T() const { return Base::get(); }
};

/*!
 **************************************************************************
 *
 * \brief  Max reducer class template for use in tbb execution.
 *
 **************************************************************************
 */
template <typename T, template <typename, typename> class Combiner>
class BaseReduceMax : public BaseReduce<T, RAJA::reduce::max, Combiner>
{
public:
  using Base = BaseReduce<T, RAJA::reduce::max, Combiner>;
  using Base::Base;

  //! reducer function; updates the current instance's state
  const BaseReduceMax &max(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }
};

/*!
 **************************************************************************
 *
 * \brief  Sum reducer class template for use in tbb execution.
 *
 **************************************************************************
 */
template <typename T, template <typename, typename> class Combiner>
class BaseReduceSum : public BaseReduce<T, RAJA::reduce::sum, Combiner>
{
public:
  using Base = BaseReduce<T, RAJA::reduce::sum, Combiner>;
  using Base::Base;

  //! reducer function; updates the current instance's state
  const BaseReduceSum &operator+=(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }
};

/*!
 **************************************************************************
 *
 * \brief  MaxLoc reducer class template for use in tbb execution.
 *
 **************************************************************************
 */
template <typename T, template <typename, typename> class Combiner>
class BaseReduceMaxLoc
    : public BaseReduce<ValueLoc<T, false>, RAJA::reduce::max, Combiner>
{
public:
  using Base = BaseReduce<ValueLoc<T, false>, RAJA::reduce::max, Combiner>;
  using value_type = typename Base::value_type;
  using Base::Base;

  //! constructor requires a default value for the reducer
  explicit BaseReduceMaxLoc(T init_val, Index_type init_idx)
      : Base(value_type(init_val, init_idx))
  {
  }

  //! reducer function; updates the current instance's state
  const BaseReduceMaxLoc &maxloc(T rhs, Index_type loc) const
  {
    this->combine(value_type(rhs, loc));
    return *this;
  }

  //! Get the calculated reduced value
  Index_type getLoc() const { return Base::get().getLoc(); }

  //! Get the calculated reduced value
  operator T() const { return Base::get(); }
};

} /* detail */

} /* reduce */

} /* RAJA */

#endif /* RAJA_PATTERN_DETAIL_REDUCE_HPP */
