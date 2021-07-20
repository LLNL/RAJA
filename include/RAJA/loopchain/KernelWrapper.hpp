//Contains class definitions and constructor functions for the kernel wrapper object

#ifndef RAJA_KernelWrapper_HPP
#define RAJA_KernelWrapper_HPP

#include "RAJA/config.hpp"
#include "RAJA/loopchain/SymExec.hpp"
#include "RAJA/pattern/kernel.hpp"

#include <vector>
#include <string>
namespace RAJA
{


//KernelWrapper wraps a kernel execution so it can be transformed before execution
//The type and constructor parameters are the same as the kernel function.
template <typename KernelPol, typename SegmentTuple, typename... Bodies>
struct KernelWrapper {
  using KPol = KernelPol;
  using BodyTuple = camp::tuple<Bodies...>;

  //these fields come from the kernel function
  const SegmentTuple segments;
  const BodyTuple bodies;
 
  //these fields are extra. they exist to enable runtime transformation
  // instead of compile time, like tile sizes.
  std::vector<camp::idx_t> overlapAmounts;
  std::vector<camp::idx_t> tileSizes;
  std::vector<SymAccess> accesses;
  static constexpr int numArgs = camp::tuple_size<SegmentTuple>::value;

  KernelWrapper(SegmentTuple  _segments, const Bodies&... _bodies) : 
    segments(_segments), bodies(_bodies...) {
     overlapAmounts = std::vector<camp::idx_t>();
     tileSizes = std::vector<camp::idx_t>();
     accesses = _execute_symbolically();
  }
  
  KernelWrapper(const KernelWrapper &) = default;
  KernelWrapper(KernelWrapper &&) = default;
  KernelWrapper & operator=(const KernelWrapper &) = default;
  KernelWrapper & operator=(KernelWrapper &&) = default;


  template <camp::idx_t Idx>
  RAJA_INLINE
  SymIterator make_sym_iterator() {
    std::string iteratorName = "i" + std::to_string(Idx);
    return SymIterator(iteratorName);
  }

  template <camp::idx_t... Is>
  RAJA_INLINE
  auto make_iterator_tuple(camp::idx_seq<Is...>) {
    auto iterators = camp::make_tuple((make_sym_iterator<Is>())...);
    return iterators;
  }
  
  std::vector<SymAccess> collect_accesses() {
    return std::vector<SymAccess>();
  }

  template <typename... Iterators>
  RAJA_INLINE
  std::vector<SymAccess> collect_accesses(SymIterator iterator, Iterators&&... rest) {
    std::vector<SymAccess> accesses = collect_accesses(std::forward<Iterators>(rest)...);

    for(long unsigned int i = 0; i < iterator.accesses->size(); i++) {
      accesses.push_back(iterator.accesses->at(i));
    }
 
    return accesses;
  }

  template <typename T, camp::idx_t... Is>
  RAJA_INLINE
  auto collect_accesses_from_iterators(T iterators, camp::idx_seq<Is...>) {
    return collect_accesses(camp::get<Is>(iterators)...);
  }

  template <typename F, typename T, camp::idx_t...Is>
  void es_helper(F function, T iterators, camp::idx_seq<Is...>) {
    function(camp::get<Is>(iterators)...);
  }  
  std::vector<SymAccess> execute_symbolically() {
    return accesses;
  }
  std::vector<SymAccess> _execute_symbolically() {
    auto iterators = make_iterator_tuple(camp::make_idx_seq_t<numArgs>());

    auto func = camp::get<0>(bodies);

    es_helper(func, iterators, camp::make_idx_seq_t<numArgs>());

    auto _accesses = collect_accesses_from_iterators(iterators, camp::make_idx_seq_t<numArgs>());
 
    return _accesses;
  }

  // Traditional execution. For normal kernels, this resolves to a call to kernel.
  // If the tile sizes or overlap tile amounts are specified at runtime,
  //  those values are added to the loop data before executing.
  template <camp::idx_t... Is>
  RAJA_INLINE
  void execute(camp::idx_seq<Is...>) const { 
  
    if(overlapAmounts.size() != 0 && tileSizes.size() != 0) {
      util::PluginContext context{util::make_context<KernelPol>()};
     
      using segment_tuple_t = typename IterableWrapperTuple<camp::decay<SegmentTuple>>::type;

      auto params = RAJA::make_tuple();
      using param_tuple_t = camp::decay<decltype(params)>;

      using loop_data_t = internal::LoopData<segment_tuple_t, param_tuple_t, camp::decay<Bodies>...>;
   
      loop_data_t loop_data(overlapAmounts, tileSizes, 
                            make_wrapped_tuple(segments), params, camp::get<Is>(bodies)...);

      util::callPostCapturePlugins(context);

      using loop_types_t = internal::makeInitialLoopTypes<loop_data_t>;

      util::callPreLaunchPlugins(context);

      RAJA_FORCEINLINE_RECURSIVE
      internal::execute_statement_list<KernelPol, loop_types_t>(loop_data);

      util::callPostLaunchPlugins(context);
    } else {
      RAJA::kernel<KernelPol>(segments, camp::get<Is>(bodies)...);
    }

  } //execute
  
  
  
  RAJA_INLINE
  void operator() () const {
    if constexpr (numArgs == 1) {
      using ForType = camp::first<KPol>;
      using StatementTypes = typename ForType::enclosed_statements_t;
      using StatementType = camp::first<StatementTypes>;
      
      if constexpr (std::is_same<StatementType,statement::TiledLambda<0>>::value) {
        auto seq = camp::make_idx_seq_t<sizeof...(Bodies)>{};
        execute(seq);
      } else {

        using ExecPol = typename ForType::execution_policy_t;
        RAJA::forall<ExecPol>(camp::get<0>(segments), camp::get<0>(bodies));
      }
    } else {
      auto seq = camp::make_idx_seq_t<sizeof...(Bodies)>{};
      execute(seq);
    }
  }

  RAJA_INLINE
  void operator() (SegmentTuple segs) {
    //TODO: Enable the kernel to be executed with a different segment
  }


  template <camp::idx_t I, camp::idx_t...Is>
  std::string segment_string_helper(camp::idx_seq<I, Is...>) {
    auto currSeg = camp::get<I>(segments);
    std::stringstream s;
    s << "(" << *currSeg.begin() << "," << *currSeg.end() << ") ";

    if constexpr (sizeof...(Is) == 0) {
      return s.str();
    } else {
      return s.str() + segment_string_helper(camp::idx_seq<Is...>{});
    }
  }

  std::string segment_string() {

    return segment_string_helper(idx_seq_for(segments));
  }


}; // KernelWrapper

//creates a kernel object using the same interface as the kernel function
template <typename KernelPol, typename SegmentTuple, typename... Bodies>
KernelWrapper<KernelPol, SegmentTuple, Bodies...> 
make_kernel(const SegmentTuple & segment,   Bodies const &... bodies) {
  return KernelWrapper<KernelPol,SegmentTuple,Bodies...>(segment, bodies...);
}


//creates a kernel object using the same interface as the forall function
template <typename ExecPol, typename Segment, typename Body> 
auto make_forall(Segment segment, const Body & body) {
  using KernPol = 
    RAJA::KernelPolicy<
      statement::For<0,ExecPol,
        statement::Lambda<0>
      >
    >;     
  
  return KernelWrapper<KernPol, camp::tuple<Segment>, Body>(camp::make_tuple(segment), body);
}

template <typename KernPol, typename SegmentTuple, typename...Bodies>
auto change_segment_tuple(KernelWrapper<KernPol, SegmentTuple, Bodies...> knl, SegmentTuple newSeg) {
  
  return make_kernel<KernPol>(newSeg, camp::get<0>(knl.bodies));
}


} //namespace RAJA


#endif

