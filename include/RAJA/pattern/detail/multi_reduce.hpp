/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief  Base types used in common for RAJA reducer objects.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_PATTERN_DETAIL_MULTI_REDUCE_HPP
#define RAJA_PATTERN_DETAIL_MULTI_REDUCE_HPP

#include "RAJA/util/Operators.hpp"
#include "RAJA/util/types.hpp"


#define RAJA_DECLARE_MULTI_REDUCER(OP_NAME, OP, POL, DATA)                               \
  template <typename T>                                                                  \
  class MultiReduce##OP_NAME<POL, T>                                                     \
      : public reduce::detail::BaseMultiReduce##OP_NAME<DATA<T, RAJA::reduce::OP<T>>>    \
  {                                                                                      \
  public:                                                                                \
    using policy = POL;                                                                  \
    using Base = reduce::detail::BaseMultiReduce##OP_NAME<DATA<T, RAJA::reduce::OP<T>>>; \
    using Base::Base;                                                                    \
    using typename Base::value_type;                                                     \
    using typename Base::reference;                                                      \
                                                                                         \
    reference operator[](size_t bin) const                                               \
    {                                                                                    \
      return reference(*this, bin);                                                      \
    }                                                                                    \
  };

#define RAJA_DECLARE_ALL_MULTI_REDUCERS(POL, DATA)            \
  RAJA_DECLARE_MULTI_REDUCER(Sum, sum, POL, DATA)             \
  RAJA_DECLARE_MULTI_REDUCER(Min, min, POL, DATA)             \
  RAJA_DECLARE_MULTI_REDUCER(Max, max, POL, DATA)             \
  RAJA_DECLARE_MULTI_REDUCER(BitOr, or_bit, POL, DATA)        \
  RAJA_DECLARE_MULTI_REDUCER(BitAnd, and_bit, POL, DATA)

namespace RAJA
{

template < typename T >
struct repeat_view
{
  struct iterator
  {
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using reference = value_type const&;

    iterator() = default;

    constexpr iterator(const T* base, size_t index)
      : m_value(base), m_index(index)
    { }

    constexpr reference operator*() const noexcept { return *m_value; }
    constexpr reference operator[](difference_type index) const noexcept { return *(*this + index); }

    constexpr iterator& operator++() { ++m_index; return *this; }
    constexpr iterator operator++(int) { auto tmp = *this; ++(*this); return tmp; }

    constexpr iterator& operator--() { --m_index; return *this; }
    constexpr iterator operator--(int) { auto tmp = *this; --(*this); return tmp; }

    constexpr iterator& operator+=(difference_type rhs) { m_index += rhs; return *this; }
    constexpr iterator& operator-=(difference_type rhs) { m_index -= rhs; return *this; }

    friend constexpr iterator operator+(iterator lhs, difference_type rhs)
    { lhs += rhs; return lhs; }
    friend constexpr iterator operator+(difference_type lhs, iterator rhs)
    { rhs += lhs; return rhs; }

    friend constexpr iterator operator-(iterator lhs, difference_type rhs)
    { lhs -= rhs; return lhs; }
    friend constexpr difference_type operator-(iterator const& lhs, iterator const& rhs)
    { return static_cast<difference_type>(lhs.m_index) - static_cast<difference_type>(rhs.m_index); }

    friend constexpr bool operator==(iterator const& lhs, iterator const& rhs)
    { return lhs.m_index == rhs.m_index; }
    friend constexpr bool operator!=(iterator const& lhs, iterator const& rhs)
    { return !(lhs == rhs); }

    friend constexpr bool operator<(iterator const& lhs, iterator const& rhs)
    { return lhs.m_index < rhs.m_index; }
    friend constexpr bool operator<=(iterator const& lhs, iterator const& rhs)
    { return !(rhs < lhs); }
    friend constexpr bool operator>(iterator const& lhs, iterator const& rhs)
    { return rhs < lhs; }
    friend constexpr bool operator>=(iterator const& lhs, iterator const& rhs)
    { return !(lhs < rhs); }

  private:
    const T* m_value = nullptr;
    size_t m_index = 0;
  };

  repeat_view() = delete;

  constexpr explicit repeat_view(T const& value, size_t bound)
    : m_bound(bound), m_value(value)
  { }

  constexpr explicit repeat_view(T&& value, size_t bound)
    : m_bound(bound), m_value(std::move(value))
  { }

  constexpr T const& front() const { return m_value; }
  constexpr T const& back() const { return m_value; }
  constexpr T const& operator[](size_t RAJA_UNUSED_ARG(index)) const { return m_value; }

  constexpr iterator begin() const { return iterator(&m_value, 0); }
  constexpr iterator cbegin() const { return iterator(&m_value, 0); }

  constexpr iterator end() const { return iterator(&m_value, m_bound); }
  constexpr iterator cend() const { return iterator(&m_value, m_bound); }

  constexpr explicit operator bool() const { return m_bound != 0; }
  constexpr bool empty() const { return m_bound == 0; }

  constexpr size_t size() const { return m_bound; }

private:
  size_t m_bound = 0;
  T m_value;
};

namespace reduce
{

namespace detail
{

template <typename t_MultiReduceData>
struct BaseMultiReduce
{
  using MultiReduceData = t_MultiReduceData;
  using MultiReduceOp = typename t_MultiReduceData::MultiReduceOp;
  using value_type = typename t_MultiReduceData::value_type;

  BaseMultiReduce() : BaseMultiReduce{repeat_view<value_type>(MultiReduceOp::identity(), 0)} {}

  explicit BaseMultiReduce(size_t num_bins,
                           value_type init_val = MultiReduceOp::identity(),
                           value_type identity = MultiReduceOp::identity())
      : BaseMultiReduce{repeat_view<value_type>(init_val, num_bins), identity}
  { }

  template < typename Container,
             concepts::enable_if_t<type_traits::is_range<Container>>* = nullptr >
  explicit BaseMultiReduce(Container const& container,
                           value_type identity = MultiReduceOp::identity())
      : data{container, identity}
  { }

  BaseMultiReduce(const BaseMultiReduce &copy) = default;
  BaseMultiReduce(BaseMultiReduce &&copy) = default;
  BaseMultiReduce &operator=(const BaseMultiReduce &) = delete;
  BaseMultiReduce &operator=(BaseMultiReduce &&) = delete;

  void reset()
  {
    reset(repeat_view<value_type>(MultiReduceOp::identity(), size()));
  }

  void reset(size_t num_bins,
             value_type init_val = MultiReduceOp::identity(),
             value_type identity = MultiReduceOp::identity())
  {
    reset(repeat_view<value_type>(init_val, num_bins), identity);
  }

  template < typename Container,
             concepts::enable_if_t<type_traits::is_range<Container>>* = nullptr >
  void reset(Container const& container,
             value_type identity = MultiReduceOp::identity())
  {
    for (size_t bin = 0; bin < data.num_bins(); ++bin) {
      RAJA_UNUSED_VAR(get(bin)); // automatic get() before reset
    }
    data.reset(container, identity);
  }

  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE
  size_t size() const { return data.num_bins(); }

  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE
  void combine(size_t bin, value_type const &other) const { data.combine(bin, other); }

  //! Get the calculated reduced value for a bin
  value_type get(size_t bin) const { return data.get(bin); }

  //! Get the calculated reduced value for each bin and store it in container
  template < typename Container,
             concepts::enable_if_t<type_traits::is_range<Container>>* = nullptr >
  void get_all(Container& container) const
  {
    size_t bin = 0;
    for (auto& val : container) {
      val = data.get(bin);
      ++bin;
    }
  }

private:
  MultiReduceData mutable data;
};


/*!
 ******************************************************************************
 *
 * \brief  Min reducer class template.
 *
 ******************************************************************************
 */
template <typename MultiReduceData>
class BaseMultiReduceMin : public BaseMultiReduce<MultiReduceData>
{
public:
  using Base = BaseMultiReduce<MultiReduceData>;
  using typename Base::value_type;
  using Base::Base;

  struct reference
  {
    RAJA_HOST_DEVICE
    reference(BaseMultiReduceMin const& base, size_t bin)
      : m_base(base), m_bin(bin)
    { }

    //! reducer function; updates the current instance's state
    RAJA_HOST_DEVICE
    reference const& min(value_type rhs) const
    {
      m_base.combine(m_bin, rhs);
      return *this;
    }

    value_type get() const
    {
      return m_base.get(m_bin);
    }

  private:
    BaseMultiReduceMin const& m_base;
    size_t m_bin;
  };
};

/*!
 **************************************************************************
 *
 * \brief  Max reducer class template.
 *
 **************************************************************************
 */
template <typename MultiReduceData>
class BaseMultiReduceMax : public BaseMultiReduce<MultiReduceData>
{
public:
  using Base = BaseMultiReduce<MultiReduceData>;
  using typename Base::value_type;

  using Base::Base;

  struct reference
  {
    RAJA_HOST_DEVICE
    reference(BaseMultiReduceMax const& base, size_t bin)
      : m_base(base), m_bin(bin)
    { }

    //! reducer function; updates the current instance's state
    RAJA_HOST_DEVICE
    reference const& max(value_type rhs) const
    {
      m_base.combine(m_bin, rhs);
      return *this;
    }

    value_type get() const
    {
      return m_base.get(m_bin);
    }

  private:
    BaseMultiReduceMax const& m_base;
    size_t m_bin;
  };
};

/*!
 **************************************************************************
 *
 * \brief  Sum reducer class template.
 *
 **************************************************************************
 */
template <typename MultiReduceData>
class BaseMultiReduceSum : public BaseMultiReduce<MultiReduceData>
{
public:
  using Base = BaseMultiReduce<MultiReduceData>;
  using typename Base::value_type;

  using Base::Base;

  struct reference
  {
    RAJA_HOST_DEVICE
    reference(BaseMultiReduceSum const& base, size_t bin)
      : m_base(base), m_bin(bin)
    { }

    //! reducer function; updates the current instance's state
    RAJA_HOST_DEVICE
    reference const& operator+=(value_type rhs) const
    {
      m_base.combine(m_bin, rhs);
      return *this;
    }

    value_type get() const
    {
      return m_base.get(m_bin);
    }

  private:
    BaseMultiReduceSum const& m_base;
    size_t m_bin;
  };
};

/*!
 **************************************************************************
 *
 * \brief  Bitwise OR reducer class template.
 *
 **************************************************************************
 */
template <typename MultiReduceData>
class BaseMultiReduceBitOr : public BaseMultiReduce<MultiReduceData>
{
public:
  using Base = BaseMultiReduce<MultiReduceData>;
  using typename Base::value_type;

  using Base::Base;

  struct reference
  {
    RAJA_HOST_DEVICE
    reference(BaseMultiReduceBitOr const& base, size_t bin)
      : m_base(base), m_bin(bin)
    { }

    //! reducer function; updates the current instance's state
    RAJA_HOST_DEVICE
    reference const& operator|=(value_type rhs) const
    {
      m_base.combine(m_bin, rhs);
      return *this;
    }

    value_type get() const
    {
      return m_base.get(m_bin);
    }

  private:
    BaseMultiReduceBitOr const& m_base;
    size_t m_bin;
  };
};

/*!
 **************************************************************************
 *
 * \brief  Bitwise AND reducer class template.
 *
 **************************************************************************
 */
template <typename MultiReduceData>
class BaseMultiReduceBitAnd : public BaseMultiReduce<MultiReduceData>
{
public:
  using Base = BaseMultiReduce<MultiReduceData>;
  using typename Base::value_type;

  using Base::Base;

  struct reference
  {
    RAJA_HOST_DEVICE
    reference(BaseMultiReduceBitAnd const& base, size_t bin)
      : m_base(base), m_bin(bin)
    { }

    //! reducer function; updates the current instance's state
    RAJA_HOST_DEVICE
    reference const& operator&=(value_type rhs) const
    {
      m_base.combine(m_bin, rhs);
      return *this;
    }

    value_type get() const
    {
      return m_base.get(m_bin);
    }

  private:
    BaseMultiReduceBitAnd const& m_base;
    size_t m_bin;
  };
};

}  // namespace detail

}  // namespace reduce

}  // namespace RAJA

#endif /* RAJA_PATTERN_DETAIL_MULTI_REDUCE_HPP */
