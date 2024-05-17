/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for
 *          sequential execution.
 *
 *          These methods should work on any platform.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_sequential_multi_reduce_HPP
#define RAJA_sequential_multi_reduce_HPP

#include "RAJA/config.hpp"

#include "RAJA/internal/MemUtils_CPU.hpp"

#include "RAJA/pattern/detail/multi_reduce.hpp"
#include "RAJA/pattern/multi_reduce.hpp"

#include "RAJA/policy/sequential/policy.hpp"

#include "RAJA/util/types.hpp"

namespace RAJA
{

namespace detail
{

/*!
 **************************************************************************
 *
 * \brief  Seq multi-reduce data class template.
 *
 * In this class memory is owned by the
 *
 **************************************************************************
 */
template < typename T, typename t_MultiReduceOp >
struct MultiReduceDataSeq
{
  using value_type = T;
  using MultiReduceOp = t_MultiReduceOp;

  MultiReduceDataSeq() = delete;

  template < typename Container,
             std::enable_if_t<!std::is_same<Container, MultiReduceDataSeq>::value>* = nullptr >
  MultiReduceDataSeq(Container const& container, T identity)
      : m_num_bins(container.size())
      , m_identity(identity)
      , m_data{create_data(container, m_num_bins)}
  { }

  MultiReduceDataSeq(MultiReduceDataSeq const &other)
      : m_parent(other.m_parent ? other.m_parent : &other)
      , m_num_bins(other.m_num_bins)
      , m_identity(other.m_identity)
      , m_data(other.m_data)
  { }

  ~MultiReduceDataSeq()
  {
    if (m_data) {
      if (!m_parent) {
        destroy_data(m_data, m_num_bins);
      }
    }
  }

  template < typename Container >
  void reset(Container const& container, T identity)
  {
    m_identity = identity;
    size_t new_num_bins = container.size();
    if (new_num_bins != m_num_bins) {
      destroy_data(m_data, m_num_bins);
      m_num_bins = new_num_bins;
      m_data = create_data(container, m_num_bins);
    } else {
      size_t bin = 0;
      for (auto const& value : container) {
        m_data[bin] = value;
        ++bin;
      }
    }
  }

  size_t num_bins() const { return m_num_bins; }

  T identity() const { return m_identity; }

  void combine(size_t bin, T const &val) { MultiReduceOp{}(m_data[bin], val); }

  T get(size_t bin) const { return m_data[bin]; }

private:
  MultiReduceDataSeq const *m_parent = nullptr;
  size_t m_num_bins;
  T m_identity;
  T* m_data;

  template < typename Container >
  static T* create_data(Container const& container, size_t num_bins)
  {
    auto data = static_cast<T*>(malloc(num_bins*sizeof(T)));
    size_t bin = 0;
    for (auto const& value : container) {
      new(&data[bin]) T(value);
      ++bin;
    }
    return data;
  }

  static void destroy_data(T*& data, size_t num_bins)
  {
    for (size_t bin = 0; bin < num_bins; ++bin) {
      data[bin].~T();
    }
    free(data);
  }
};

}  // namespace detail

RAJA_DECLARE_ALL_MULTI_REDUCERS(seq_multi_reduce, detail::MultiReduceDataSeq)

}  // namespace RAJA

#endif  // closing endif for header file include guard
