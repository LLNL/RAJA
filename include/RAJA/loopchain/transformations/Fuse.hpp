// Contains code that implements the fuse transformation for kernels
// Because this transformation does not analyze dependences and just applie
// the transformation, this file does not include any ISL analysis code
#ifndef RAJA_LCFUSE_HPP
#define RAJA_LCFUSE_HPP

#include "RAJA/config.hpp"
#include "RAJA/loopchain/KernelWrapper.hpp"
#include "RAJA/loopchain/Chain.hpp"
#include "RAJA/loopchain/transformations/Common.hpp"
#include "RAJA/loopchain/transformations/BoundaryKernels.hpp"

namespace RAJA
{

//Transformation Declarations

template <typename... KernelTypes>
auto fuse(camp::tuple<KernelTypes...> knlTuple);

template <typename...KernelTypes>
auto fuse(KernelTypes... knls);


//Transformation Definitions

template <typename...LambdaTypes, camp::idx_t I0, camp::idx_t I1>
RAJA_INLINE
auto fused_lambda(camp::tuple<LambdaTypes...> lambdas, camp::idx_seq<I0,I1>) {

  return [=](auto...is) {
    camp::get<I0>(lambdas)(is...);
    camp::get<I1>(lambdas)(is...);
    };
}

//specialized lambdas for 4 and 11 kernel fusions
template <typename...LambdaTypes, camp::idx_t I0, camp::idx_t I1, camp::idx_t I2, camp::idx_t I3>
RAJA_INLINE
auto fused_lambda(camp::tuple<LambdaTypes...> lambdas, camp::idx_seq<I0,I1,I2,I3>) {

  return [=](auto...is) {
    camp::get<I0>(lambdas)(is...);
    camp::get<I1>(lambdas)(is...);
    camp::get<I2>(lambdas)(is...);
    camp::get<I3>(lambdas)(is...);
  };
}

template <typename...LambdaTypes, camp::idx_t I0, camp::idx_t I1, camp::idx_t I2, 
          camp::idx_t I3, camp::idx_t I4, camp::idx_t I5, camp::idx_t I6, 
          camp::idx_t I7, camp::idx_t I8, camp::idx_t I9, camp::idx_t I10>
RAJA_INLINE
auto fused_lambda(camp::tuple<LambdaTypes...> lambdas, camp::idx_seq<I0,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10>) {

  return [=](auto...is) {
    camp::get<I0>(lambdas)(is...);
    camp::get<I1>(lambdas)(is...);
    camp::get<I2>(lambdas)(is...);
    camp::get<I3>(lambdas)(is...);
    camp::get<I4>(lambdas)(is...);
    camp::get<I5>(lambdas)(is...);
    camp::get<I6>(lambdas)(is...);
    camp::get<I7>(lambdas)(is...);
    camp::get<I8>(lambdas)(is...);
    camp::get<I9>(lambdas)(is...);
    camp::get<I10>(lambdas)(is...);
  };
}

template <typename...LambdaTypes, camp::idx_t I0, camp::idx_t I1, camp::idx_t I2,
          camp::idx_t I3, camp::idx_t I4, camp::idx_t I5, camp::idx_t I6,
          camp::idx_t I7, camp::idx_t I8, camp::idx_t I9, camp::idx_t I10, camp::idx_t I11>
RAJA_INLINE
auto fused_lambda(camp::tuple<LambdaTypes...> lambdas, camp::idx_seq<I0,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11>) {

  return [=](auto...is) {
    camp::get<I0>(lambdas)(is...);
    camp::get<I1>(lambdas)(is...);
    camp::get<I2>(lambdas)(is...);
    camp::get<I3>(lambdas)(is...);
    camp::get<I4>(lambdas)(is...);
    camp::get<I5>(lambdas)(is...);
    camp::get<I6>(lambdas)(is...);
    camp::get<I7>(lambdas)(is...);
    camp::get<I8>(lambdas)(is...);
    camp::get<I9>(lambdas)(is...);
    camp::get<I10>(lambdas)(is...);
    camp::get<I11>(lambdas)(is...);
  };
}

// Terminal case for recursion, 0 left
template <typename Args, camp::idx_t...Is>
RAJA_INLINE
void fl_helper(Args&&, camp::idx_seq<Is...>) {
    // Yup, done
}
template <typename Args, camp::idx_t...Is, typename L1, typename...LambdaTypes>
RAJA_INLINE
void fl_helper(Args&& args_tuple, camp::idx_seq<Is...> seq, L1&& l1, LambdaTypes&&... lrest) {
      l1(camp::get<Is>(args_tuple)...);
      fl_helper(std::forward<Args>(args_tuple), seq, std::forward<LambdaTypes>(lrest)...);
}
template <typename...LambdaTypes, camp::idx_t...Is>
RAJA_INLINE
auto fused_lambda(camp::tuple<LambdaTypes...> lambdas, camp::idx_seq<Is...>) {
  return [=](auto...is) {
      fl_helper(camp::forward_as_tuple(is...), idx_seq_from_to<0,sizeof...(is)>(), camp::get<Is>(lambdas)...);
  };

}


// returns a kernel object that executes the kernels in knlTuple in a fused schedule for the 
// parts of their iteration spaces that they all share. For each iteration point in the 
// shared iteration space, the produced kernel will execute the first kernel in the tuple for
// that iteration point, then the second kernel, etc. After each kernel has been executed for 
// that iteration point, the produced kernel moves to the second iteration point and repeats
template <typename...KernelTypes, camp::idx_t...Is>
RAJA_INLINE
auto fused_knl(camp::tuple<KernelTypes...> knlTuple, camp::idx_seq<Is...> seq) {

  auto segmentTuples = make_tuple(camp::get<Is>(knlTuple).segments...);
  auto sharedIterSpaceTuple = intersect_segment_tuples(segmentTuples);

  auto lambdaTuple = make_tuple(camp::get<0>(camp::get<Is>(knlTuple).bodies)...);
  auto fusedLambda = fused_lambda(lambdaTuple, seq);
  
  auto sampleKnl = camp::get<0>(knlTuple);
  using KPol = typename decltype(sampleKnl)::KPol;

  return make_kernel<KPol>(sharedIterSpaceTuple, fusedLambda);
}


template <typename... KernelTypes, camp::idx_t...Is>
RAJA_INLINE
auto fuse(camp::tuple<KernelTypes...> knlTuple, camp::idx_seq<Is...> seq) {
  auto lowBoundKnls = low_boundary_knls(knlTuple, seq);
  auto highBoundKnls = high_boundary_knls(knlTuple, seq);
  auto fusedKnl = fused_knl(knlTuple, seq);

  auto allKnls = tuple_cat(lowBoundKnls, make_tuple(fusedKnl), highBoundKnls);
  
  
  if(equal_iterspaces(knlTuple)) {
    return grouped_kernels(allKnls,1);
  } else {
    return grouped_kernels(allKnls,0);
  }

}


template <typename... KernelTypes>
RAJA_INLINE
auto fuse(camp::tuple<KernelTypes...> knlTuple) {
  return fuse(knlTuple, idx_seq_for(knlTuple));
}

template <typename...KernelTypes>
RAJA_INLINE
auto fuse(KernelTypes... knls) {
  return fuse(make_tuple(knls...));
}

} //namespace RAJA

#endif
