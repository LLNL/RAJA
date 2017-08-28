#ifndef RAJA_PATTERN_DETAIL_REDUCE_HPP
#define RAJA_PATTERN_DETAIL_REDUCE_HPP

#include "RAJA/pattern/reduce.hpp"
#include "RAJA/util/Operators.hpp"
#include "RAJA/util/types.hpp"

#define RAJA_DECLARE_REDUCER(OP, POL, COMBINER)                         \
  template <typename T>                                                 \
  class Reduce##OP<POL, T> : public detail::BaseReduce##OP<T, COMBINER> \
  {                                                                     \
  public:                                                               \
    using Base = detail::BaseReduce##OP<T, COMBINER>;                   \
    using Base::Base;                                                   \
  };

#define RAJA_DECLARE_ALL_REDUCERS(POL, COMBINER) \
  RAJA_DECLARE_REDUCER(Sum, POL, COMBINER)       \
  RAJA_DECLARE_REDUCER(Min, POL, COMBINER)       \
  RAJA_DECLARE_REDUCER(Max, POL, COMBINER)       \
  RAJA_DECLARE_REDUCER(MinLoc, POL, COMBINER)    \
  RAJA_DECLARE_REDUCER(MaxLoc, POL, COMBINER)

namespace RAJA
{

namespace detail
{

template <typename T, bool doing_min = true>
class ValueLoc
{
public:
  T val = doing_min ? operators::limits<T>::max() : operators::limits<T>::min();
  Index_type loc = -1;

  constexpr ValueLoc() = default;
  constexpr ValueLoc(ValueLoc const &) = default;
  ValueLoc &operator=(ValueLoc const &) = default;
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
}  // end namespace detail

namespace operators
{
template <typename T, bool B>
struct limits<::RAJA::detail::ValueLoc<T, B>> : limits<T> {
};
}

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

  void combine(T const &other) const
  {
    c.combine(other);
  }

  T &local() const { return c.local(); }

  //! Get the calculated reduced value
  operator T() const { return c.get(); }

  //! Get the calculated reduced value
  T get() const { return c.get(); }
};

template <typename T, typename Reduce, typename Derived>
class BaseCombinable
{
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

  ~BaseCombinable() { derived().destructing(); }

  void combine(T const &other) { Reduce{}(my_data, other); }

  /*!
   *  \return the calculated reduced value
   */
  T get() const { return derived().get_combined(); }

  /*!
   *  \return reference to the local value
   */
  T &local() { return my_data; }

protected:
  void constructing() {}
  void destructing()
  {
    if (parent) {
      Reduce()(parent->my_data, my_data);
    }
  }
  T get_combined() { return my_data; }

private:
  // Convenience method for CRTP
  Derived &derived() { return *static_cast<Derived *>(this); }
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

} /* RAJA */

#endif /* RAJA_PATTERN_DETAIL_REDUCE_HPP */
