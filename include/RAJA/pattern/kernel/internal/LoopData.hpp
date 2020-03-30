/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for loop kernel internals: LoopData structure and
 *          related helper functions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_kernel_internal_LoopData_HPP
#define RAJA_pattern_kernel_internal_LoopData_HPP

#include "RAJA/config.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "camp/camp.hpp"

#include "RAJA/pattern/detail/privatizer.hpp"
#include "RAJA/pattern/kernel/internal/StatementList.hpp"

#include <iterator>
#include <type_traits>

namespace RAJA
{
namespace internal
{




  // Universal base of all For wrappers for type traits
  struct ForList {
  };
  struct ForBase {
  };
  struct CollapseBase {
  };
  template <camp::idx_t ArgumentId, typename Policy>
  struct ForTraitBase : public ForBase {
    constexpr static camp::idx_t index_val = ArgumentId;
    using index = camp::num<ArgumentId>;
    using index_type = camp::nil;  // default to invalid type
    using policy_type = Policy;
    using type = ForTraitBase;  // make camp::value compatible
  };




template <typename Iterator>
struct iterable_difftype_getter {
  using type = typename std::iterator_traits<
      typename Iterator::iterator>::difference_type;
};

template <typename Segments>
using difftype_list_from_segments =
    typename camp::transform<iterable_difftype_getter, Segments>::type;


template <typename Segments>
using difftype_tuple_from_segments =
    typename camp::apply_l<camp::lambda<camp::tuple>,
                           difftype_list_from_segments<Segments>>::type;


template <typename Iterator>
struct iterable_value_type_getter {
  using type =
      typename std::iterator_traits<typename Iterator::iterator>::value_type;
};

template <typename Segments>
using value_type_list_from_segments =
    typename camp::transform<iterable_value_type_getter, Segments>::type;


template <typename Segments>
using index_tuple_from_segments =
    typename camp::apply_l<camp::lambda<camp::tuple>,
                           value_type_list_from_segments<Segments>>::type;




template <typename SegmentTuple,
          typename ParamTuple,
          typename... Bodies>
struct LoopData {

  using Self = LoopData<SegmentTuple, ParamTuple, Bodies...>;

  using offset_tuple_t =
      difftype_tuple_from_segments<typename SegmentTuple::TList>;

  using index_tuple_t = index_tuple_from_segments<typename SegmentTuple::TList>;


  using segment_tuple_t = SegmentTuple;
  SegmentTuple segment_tuple;

  using param_tuple_t = ParamTuple;
  ParamTuple param_tuple;

  using BodiesTuple = camp::tuple<Bodies...>;
  const BodiesTuple bodies;
  offset_tuple_t offset_tuple;

  RAJA_INLINE
  LoopData(SegmentTuple const &s, ParamTuple const &p, Bodies const &... b)
      : segment_tuple(s), param_tuple(p), bodies(b...)
  {
    //assign_begin_all();
  }

  template <typename SegmentTuple0,
            typename ParamTuple0,
            typename... Bodies0>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr LoopData(
      LoopData<SegmentTuple0, ParamTuple0, Bodies0...> &c)
      : segment_tuple(c.segment_tuple),
        param_tuple(c.param_tuple),
        bodies(c.bodies),
        offset_tuple(c.offset_tuple)
  {
  }

  template <camp::idx_t Idx, typename IndexT>
  RAJA_HOST_DEVICE RAJA_INLINE void assign_offset(IndexT const &i)
  {
    camp::get<Idx>(offset_tuple) = i;
  }

  template <typename ParamId, typename IndexT>
  RAJA_HOST_DEVICE RAJA_INLINE void assign_param(IndexT const &i)
  {
    using param_t = camp::at_v<typename param_tuple_t::TList, ParamId::param_idx>;
    camp::get<ParamId::param_idx>(param_tuple) = param_t(i);
  }

  template <typename ParamId>
  RAJA_HOST_DEVICE RAJA_INLINE
  auto get_param() ->
    camp::at_v<typename param_tuple_t::TList, ParamId::param_idx>
  {
    return camp::get<ParamId::param_idx>(param_tuple);
  }


};




template <camp::idx_t ArgumentId, typename Data>
RAJA_INLINE RAJA_HOST_DEVICE auto segment_length(Data const &data) ->
    typename std::iterator_traits<
        typename camp::at_v<typename Data::segment_tuple_t::TList,
                            ArgumentId>::iterator>::difference_type
{
  return camp::get<ArgumentId>(data.segment_tuple).end() -
         camp::get<ArgumentId>(data.segment_tuple).begin();
}




template <typename Data, typename Types, typename... EnclosedStmts>
struct GenericWrapper : GenericWrapperBase {
  using data_t = camp::decay<Data>;

  data_t &data;

  RAJA_INLINE
  constexpr explicit GenericWrapper(data_t &d) : data{d} {}

  RAJA_INLINE
  void exec() { execute_statement_list<camp::list<EnclosedStmts...>, Types>(data); }
};


/*!
 * Convenience object used to create thread-private a LoopData object.
 */
template <typename T>
struct NestedPrivatizer {
  using data_t = typename T::data_t;
  using value_type = camp::decay<T>;
  using reference_type = value_type &;

  data_t privatized_data;
  value_type privatized_wrapper;

  RAJA_INLINE
  constexpr NestedPrivatizer(const T &o)
      : privatized_data{o.data}, privatized_wrapper(privatized_data)
  {
  }

  RAJA_INLINE
  reference_type get_priv() { return privatized_wrapper; }
};



}  // end namespace internal
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_internal_LoopData_HPP */
