#ifndef RAJA_PATTERN_DETAIL_REDUCE_HPP
#define RAJA_PATTERN_DETAIL_REDUCE_HPP

#include "RAJA/pattern/reduce.hpp"
#include "RAJA/util/Operators.hpp"
#include "RAJA/util/types.hpp"

namespace RAJA
{

namespace detail
{

template <typename T, bool doing_min = true>
struct ValueLoc {
  T val = doing_min ? operators::limits<T>::max() : operators::limits<T>::min();
  Index_type loc = -1;
  constexpr ValueLoc() = default;
  constexpr ValueLoc(ValueLoc const &) = default;
  ValueLoc &operator=(ValueLoc const &) = default;
  constexpr ValueLoc(T const &val) : val{val}, loc{-1} {}
  constexpr ValueLoc(T const &val, Index_type const &loc) : val{val}, loc{loc}
  {
  }
  operator T() const { return val; }
  bool operator<(ValueLoc const &rhs) const { return val < rhs.val; }
  bool operator>(ValueLoc const &rhs) const { return val > rhs.val; }
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
class BaseReduceMin : public Combiner<T, RAJA::reduce::min<T>>
{
public:
  using Operator = RAJA::reduce::min<T>;
  using Base = Combiner<T, Operator>;

  //! prohibit compiler-generated default ctor
  BaseReduceMin() = delete;

  //! prohibit compiler-generated copy assignment
  BaseReduceMin &operator=(const BaseReduceMin &) = delete;

  //! compiler-generated copy constructor
  BaseReduceMin(const BaseReduceMin &) = default;

  //! compiler-generated move constructor
  BaseReduceMin(BaseReduceMin &&) = default;

  //! compiler-generated move assignment
  BaseReduceMin &operator=(BaseReduceMin &&) = default;

  explicit BaseReduceMin(T init_val,
                         T initializer = operators::limits<T>::max())
      : Base(init_val, initializer)
  {
  }
  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  const BaseReduceMin &min(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }

  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  BaseReduceMin &min(T rhs)
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
template <typename TT, template <typename, typename> class Combiner>
class BaseReduceMinLoc
    : public Combiner<ValueLoc<TT>, RAJA::reduce::min<ValueLoc<TT>>>
{
public:
  using T = ValueLoc<TT>;
  using Operator = RAJA::reduce::min<T>;
  using Base = Combiner<T, Operator>;
  //! prohibit compiler-generated default ctor
  BaseReduceMinLoc() = delete;

  //! prohibit compiler-generated copy assignment
  BaseReduceMinLoc &operator=(const BaseReduceMinLoc &) = delete;

  //! compiler-generated copy constructor
  BaseReduceMinLoc(const BaseReduceMinLoc &) = default;

  //! compiler-generated move constructor
  BaseReduceMinLoc(BaseReduceMinLoc &&) = default;

  //! compiler-generated move assignment
  BaseReduceMinLoc &operator=(BaseReduceMinLoc &&) = default;

  //! constructor requires a default value for the reducer
  explicit BaseReduceMinLoc(T init_val, Index_type init_idx)
      : Base(typename Base::value_type(init_val, init_idx))
  {
  }
  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  const BaseReduceMinLoc &minloc(T rhs, Index_type loc) const
  {
    this->combine(typename Base::value_type(rhs, loc));
    return *this;
  }

  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  BaseReduceMinLoc &minloc(T rhs, Index_type loc)
  {
    this->combine(typename Base::value_type(rhs, loc));
    return *this;
  }

  Index_type getLoc() { return Base::get().loc; }
  operator TT() const { return Base::get(); }
};

/*!
 **************************************************************************
 *
 * \brief  Max reducer class template for use in tbb execution.
 *
 **************************************************************************
 */
template <typename T, template <typename, typename> class Combiner>
class BaseReduceMax : public Combiner<T, RAJA::reduce::max<T>>
{
public:
  using Operator = RAJA::reduce::max<T>;
  using Base = Combiner<T, Operator>;
  using Base::Base;
  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  const BaseReduceMax &max(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }

  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  BaseReduceMax &max(T rhs)
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
class BaseReduceSum : public Combiner<T, RAJA::reduce::sum<T>>
{
public:
  using Operator = RAJA::reduce::sum<T>;
  using Base = Combiner<T, Operator>;
  using Base::Base;
  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  const BaseReduceSum &operator+=(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }

  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  BaseReduceSum &operator+=(T rhs)
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
template <typename TT, template <typename, typename> class Combiner>
class BaseReduceMaxLoc : public Combiner<ValueLoc<TT, false>,
                                         RAJA::reduce::max<ValueLoc<TT, false>>>
{
public:
  using T = ValueLoc<TT, false>;
  using Operator = RAJA::reduce::max<T>;
  using Base = Combiner<T, Operator>;
  //! prohibit compiler-generated default ctor
  BaseReduceMaxLoc() = delete;

  //! prohibit compiler-generated copy assignment
  BaseReduceMaxLoc &operator=(const BaseReduceMaxLoc &) = delete;

  //! compiler-generated copy constructor
  BaseReduceMaxLoc(const BaseReduceMaxLoc &) = default;

  //! compiler-generated move constructor
  BaseReduceMaxLoc(BaseReduceMaxLoc &&) = default;

  //! compiler-generated move assignment
  BaseReduceMaxLoc &operator=(BaseReduceMaxLoc &&) = default;

  //! constructor requires a default value for the reducer
  explicit BaseReduceMaxLoc(T init_val, Index_type init_idx)
      : Base(typename Base::value_type(init_val, init_idx))
  {
  }
  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  const BaseReduceMaxLoc &maxloc(T rhs, Index_type loc) const
  {
    this->combine(typename Base::value_type(rhs, loc));
    return *this;
  }

  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  BaseReduceMaxLoc &maxloc(T rhs, Index_type loc)
  {
    this->combine(typename Base::value_type(rhs, loc));
    return *this;
  }

  Index_type getLoc() { return Base::get().loc; }

  operator TT() const { return Base::get(); }
};

} /* detail */

} /* RAJA */

#endif /* RAJA_PATTERN_DETAIL_REDUCE_HPP */
