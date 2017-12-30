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
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_nested_internal_HPP
#define RAJA_pattern_nested_internal_HPP

#include "RAJA/config.hpp"
#include "RAJA/index/IndexSet.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

#include "camp/camp.hpp"
#include "camp/concepts.hpp"

#include <type_traits>

namespace RAJA
{
namespace nested
{
namespace internal
{


template <typename... Stmts>
using StatementList = camp::tuple<Stmts...>;


template <typename ExecPolicy, typename... EnclosedStmts>
struct Statement {
  using enclosed_statements_t = StatementList<EnclosedStmts...>;
  enclosed_statements_t enclosed_statements;

  using execution_policy_t = ExecPolicy;
};


template <typename T>
using remove_all_t =
    typename std::remove_cv<typename std::remove_reference<T>::type>::type;

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
struct iterable_value_type_getter {
  using type = typename Iterator::iterator::value_type;
};
template <>
struct iterable_value_type_getter<IndexSet> {
  // TODO: when static indexset drops, specialize properly
  using type = Index_type;
};

template <typename Segments>
using value_type_list_from_segments =
    typename camp::transform<iterable_value_type_getter, Segments>::type;


template <typename Segments>
using index_tuple_from_segments =
    typename camp::apply_l<camp::lambda<camp::tuple>,
                           value_type_list_from_segments<Segments> >::type;


template <typename Policy>
struct StatementExecutor{};


template <typename PolicyType, typename SegmentTuple, typename ... Bodies>
struct LoopData {

  using Self = LoopData<PolicyType, SegmentTuple, Bodies...>;

  using index_tuple_t = index_tuple_from_segments<typename SegmentTuple::TList>;

  const PolicyType policy;
  SegmentTuple segment_tuple;

  using BodiesTuple = camp::tuple<typename std::remove_reference<Bodies>::type...> ;
  const BodiesTuple bodies;
  index_tuple_t index_tuple;

  RAJA_INLINE
  constexpr
  LoopData(PolicyType const &p, SegmentTuple const &s, Bodies const & ... b)
      : policy{p}, segment_tuple{s}, bodies{b...}
  {
//    printf("LoopData: policy=%d, segment_tuple=%d, bodies=%d\n",
//        (int)sizeof(policy), (int)sizeof(segment_tuple), (int)sizeof(bodies));
  }

  template <camp::idx_t Idx, typename IndexT>
  RAJA_HOST_DEVICE
  void assign_index(IndexT const &i)
  {
    camp::get<Idx>(index_tuple) =
        camp::tuple_element_t<Idx, index_tuple_t>{i};
  }
};


template<camp::idx_t LoopIndex, typename Data>
RAJA_INLINE
RAJA_HOST_DEVICE
void invoke_lambda(Data &data){
  camp::invoke(data.index_tuple, camp::get<LoopIndex>(data.bodies));
}



template <camp::idx_t idx, camp::idx_t N>
struct StatementListExecutor;


template<typename StmtList, typename Data>
void execute_statement_list(StmtList && statement_list, Data && data){
  using statement_list_type = camp::decay<StmtList>;
  StatementListExecutor<0, camp::tuple_size<statement_list_type>::value> launcher;
  launcher(statement_list, std::forward<Data>(data));
}



template <typename StmtList, typename Data>
struct StatementListWrapper {

  using data_type = typename std::remove_reference<Data>::type;
  using statement_list_type = typename std::remove_reference<StmtList>::type;

  StmtList const &statement_list;
  Data &data;

  RAJA_INLINE
  constexpr
  StatementListWrapper(StmtList const &pt, Data &d) : statement_list(pt), data{d} {}

  RAJA_INLINE
  constexpr
  void operator()() const
  {
    execute_statement_list(statement_list, data);
  }
};



// Create a wrapper for this policy
template<typename PolicyT, typename Data>
RAJA_INLINE
constexpr
auto make_statement_list_wrapper(PolicyT && policy, Data && data) ->
  StatementListWrapper<decltype(policy), camp::decay<Data>>
{
  return StatementListWrapper<decltype(policy), camp::decay<Data>>(
      policy, std::forward<Data>(data));
}


template <camp::idx_t statement_index, camp::idx_t num_statements>
struct StatementListExecutor{

  template<typename StmtList, typename Data>
  RAJA_INLINE
  void operator()(StmtList const &statement_list, Data &&data) const {

    // Get the statement we're going to execute
    auto const &statement = camp::get<statement_index>(statement_list);

    // Create a wrapper for enclosed statements
    auto enclosed_wrapper = make_statement_list_wrapper(statement.enclosed_statements, std::forward<Data>(data));

    // Execute this statement
    StatementExecutor<remove_all_t<decltype(statement)>> e;
    e(statement, enclosed_wrapper);

    // call our next statement
    StatementListExecutor<statement_index+1, num_statements> next_sequential;
    next_sequential(statement_list, std::forward<Data>(data));
  }
};


/*
 * termination case, a NOP.
 */

template <camp::idx_t num_statements>
struct StatementListExecutor<num_statements,num_statements> {

  template<typename StmtList, typename Data>
  RAJA_INLINE
  constexpr
  void operator()(StmtList const &, Data &&) const {

  }

};





template <camp::idx_t Index, typename BaseWrapper>
struct GenericWrapper {
  using data_type = camp::decay<typename BaseWrapper::data_type>;
  using statement_list_type = camp::decay<typename BaseWrapper::statement_list_type>;

  BaseWrapper wrapper;

  RAJA_HOST_DEVICE
  RAJA_INLINE
  constexpr
  GenericWrapper(BaseWrapper const &w) : wrapper{w} {}

  RAJA_HOST_DEVICE
  RAJA_INLINE
  constexpr
  GenericWrapper(statement_list_type const &s, data_type &d) : wrapper{s,d} {}

};


/*!
 * Convenience object used to create thread-private a LoopData object.
 */
template <typename T>
struct NestedPrivatizer {
  using data_type = typename T::data_type;
  using value_type = camp::decay<T>;
  using reference_type = value_type &;

  data_type privatized_data;
  value_type privatized_wrapper;

  RAJA_INLINE
  constexpr
  NestedPrivatizer(const T &o) : privatized_data{o.wrapper.data}, privatized_wrapper(value_type{o.wrapper.statement_list, privatized_data}) {}

  RAJA_INLINE
  reference_type get_priv() { return privatized_wrapper; }
};

template <typename StmtList, typename Data>
struct NestedPrivatizer<StatementListWrapper<StmtList, Data>> {
  using data_type = Data;
  using value_type = StatementListWrapper<StmtList, Data>;
  using reference_type = value_type &;

  data_type privatized_data;
  value_type privatized_wrapper;

  RAJA_INLINE
  constexpr
  NestedPrivatizer(const StatementListWrapper<StmtList, Data> &wrapper) : privatized_data{wrapper.data}, privatized_wrapper(value_type{wrapper.statement_list, privatized_data}) {}

  RAJA_INLINE
  reference_type get_priv() { return privatized_wrapper; }
};

/**
 * @brief specialization of internal::thread_privatize for nested
 */
template <typename StmtList, typename Data>
constexpr
auto thread_privatize(const nested::internal::StatementListWrapper<StmtList, Data> &item)
    -> NestedPrivatizer<nested::internal::StatementListWrapper<StmtList, Data>>
{
  return NestedPrivatizer<nested::internal::StatementListWrapper<StmtList, Data>>{item};
}

}  // end namespace internal
}  // end namespace nested



#ifdef RAJA_ENABLE_CHAI

namespace detail
{


template <typename T>
struct get_statement_platform
{
  static constexpr Platform value =
          get_platform_from_list<typename T::execution_policy_t, typename T::enclosed_statements_t>::value;
};

/*!
 * Specialization to define the platform for an nested::StatementList, and
 * (by alias) a nested::Policy
 *
 * This collects the Platform from each of it's statements, recursing into
 * each of them.
 */
template <typename... Stmts>
struct get_platform<RAJA::nested::internal::StatementList<Stmts...>>{
  static constexpr Platform value =
        VarOps::foldl(max_platform(), get_statement_platform<Stmts>::value...);
};

/*!
 * Specialize for an empty statement list to be undefined
 */
template <>
struct get_platform<RAJA::nested::internal::StatementList<>>{
  static constexpr Platform value = Platform::undefined;
};



}  // end detail namespace

#endif  // RAJA_ENABLE_CHAI


}  // end namespace RAJA


#endif /* RAJA_pattern_nested_internal_HPP */
