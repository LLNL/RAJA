// Contains definitions for the types that are returned by fuse functions and such


#ifndef RAJA_Chain_HPP
#define RAJA_Chain_HPP

#include "RAJA/config.hpp"


namespace RAJA
{

// Encodes the result of grouping together a sequence of perfectly nested 
// kernels. If there were originally n d-dimensional kernels, their associated 
// GroupedKernels object will have 2*n*d + 1 kernels in it. Executing a 
// GroupedKernels object will execute the kernels in the order they were 
// provided to the constructor. MainKnlIdx is the index of the kernel whose 
// iteration space is the intersection of all the iteration spaces of the 
// original kernels. Currently, this index value is the middle value in the 
// sequence.
template <camp::idx_t MainKnlIdx, typename... KernelTypes>
struct GroupedKernels {
  using KernelTuple = camp::tuple<KernelTypes...>;

  const KernelTuple knlTuple;
  
  static constexpr camp::idx_t numKnls = sizeof...(KernelTypes);
  bool allSame;
  GroupedKernels(const KernelTuple & knlTuple_, bool allSame_) : knlTuple(knlTuple_), allSame(allSame_) {};
  GroupedKernels(const KernelTuple & knlTuple_) : knlTuple(knlTuple_), allSame(0) {};

  // returns a tuple of kernels that would execute before the main kernel
  auto pre_knls() {
    return tuple_slice<0,MainKnlIdx>(knlTuple);
  }

  // returns a tuple of kernels that would execute after the main kernel
  auto post_knls() {
    return tuple_slice<MainKnlIdx+1, numKnls>(knlTuple);
  }

  // returns the kernel that executes over the shared iteration space
  auto main_knl() {
    return camp::get<MainKnlIdx>(knlTuple);
  }

  // executes each kernel in the object one at a time
  template <camp::idx_t I, camp::idx_t...Is>  
  RAJA_INLINE
  void execute(camp::idx_seq<I, Is...>) {
    camp::get<I>(knlTuple)();
    if constexpr (sizeof...(Is) == 0) {
      return;
    } else {
      execute(camp::idx_seq<Is...>{});
    }
  }

  RAJA_INLINE
  void operator() () {
    if(allSame) {
      execute_main_knl();
    } else {
      auto seq = camp::make_idx_seq_t<numKnls>{};
      execute(seq);
    }
  }

  RAJA_INLINE
  void execute_main_knl() {
       camp::get<MainKnlIdx>(knlTuple)();
  }

}; //GroupedKernels


// constructor function for GroupedKernels that takes a tuple of kernels
template <typename...KernelTypes>
RAJA_INLINE
auto grouped_kernels(camp::tuple<KernelTypes...> knlTuple, int allSame) {
  constexpr auto numKnls = sizeof...(KernelTypes);
  constexpr camp::idx_t MainKnlIdx = (numKnls - 1) / 2;

  return GroupedKernels<MainKnlIdx, KernelTypes...>(knlTuple, allSame);
}
template <camp::idx_t MainKnlIdx, typename...KernelTypes>
RAJA_INLINE
auto grouped_kernels(camp::tuple<KernelTypes...> knlTuple) {
  return GroupedKernels<MainKnlIdx, KernelTypes...>(knlTuple, 0);
}

// constructor function for GroupedKernels that takes kernels
template <camp::idx_t MainKnlIdx, typename...KernelTypes>
RAJA_INLINE
auto grouped_kernels(KernelTypes... knls) {
  return grouped_kernels<MainKnlIdx>(camp::make_tuple(knls...));
}

template <typename...KernelTypes>
auto grouped_kernels(KernelTypes... knls) {
  constexpr auto numKnls = sizeof...(KernelTypes);

  //numKnls = 2 *nd + 1;
  constexpr auto mainIndex = (numKnls - 1) / 2;

  return grouped_kernels<mainIndex>(knls...);
}
// Encodes a sequence of loops. Executing a LoopChain object executes each 
//  kernel in the chain one at a time.
template <typename... KernelTypes>
struct LoopChain {
  using KernelTuple = camp::tuple<KernelTypes...>;
  
  const KernelTuple knlTuple;
  
  LoopChain(const KernelTuple & knlTuple_) : knlTuple(knlTuple_) {}

  template <camp::idx_t I, camp::idx_t...Is>
  void execute(camp::idx_seq<I,Is...>) {
    camp::get<I>(knlTuple)();
    if constexpr (sizeof...(Is) == 0) {
      return;
    } else {
      execute(camp::idx_seq<Is...>{});
    }
  }
}; //LoopChain

// LoopChain constructor function for kernel tuple
template <typename... KernelTypes>
auto loop_chain(camp::tuple<KernelTypes...> knlTuple) {
  return LoopChain<KernelTypes...>(knlTuple);
}

// LoopChain constructor function for kernels
template <typename...KernelTypes>
auto loop_chain(KernelTypes...knls) {
  return LoopChain<KernelTypes...>(camp::make_tuple(knls...));
}

} //namespace RAJA
#endif
