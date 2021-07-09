// Contains code for the generation of boundary kernels for fusing transformations.
// When kernels are transformed in a way that applies a fusion, we only work with
// the parts of the kernels that are the same iterator values. Thus, we need to 
// execute the other kernels, and these functions generate those.


#ifndef RAJA_LCBOUND_HPP
#define RAJA_LCBOUND_HPP

#include "RAJA/config.hpp"
#include "RAJA/loopchain/KernelWrapper.hpp"
#include "RAJA/loopchain/Chain.hpp"
#include "RAJA/loopchain/transformations/Common.hpp"
namespace RAJA
{

template <typename...KernelTypes, camp::idx_t...Is>
auto high_boundary_knls(camp::tuple<KernelTypes...> knlTuple, camp::idx_seq<Is...> seq);

template <typename...KernelTypes, camp::idx_t...Is>
auto low_boundary_knls(camp::tuple<KernelTypes...> knlTuple, camp::idx_seq<Is...> seq);


// Definitions

template < camp::idx_t I, typename SegmentTuple>
RAJA_INLINE
auto high_boundary_end_dims_helper(SegmentTuple knlSegments, SegmentTuple sharedSegments) {
  auto knlSegment = camp::get<I>(knlSegments);
  auto sharedSegment = camp::get<I>(sharedSegments);

  using RangeType = decltype(knlSegment);
  return RangeType(*sharedSegment.begin(), *knlSegment.end());
}

template <typename SegmentTuple, camp::idx_t... Is>
RAJA_INLINE
auto high_boundary_end_dims(SegmentTuple knlSegments, SegmentTuple sharedSegments, camp::idx_seq<Is...>) {
  return make_tuple((high_boundary_end_dims_helper<Is>(knlSegments, sharedSegments))...);
}


// Creates one of the hyper-rectangles in the upper boundary
// iteration space decomposition for a single kernel
template <camp::idx_t I,typename...SegmentTypes,  camp::idx_t...Is>
auto high_boundary_decomp_piece(camp::tuple<SegmentTypes...> knlSegments,
                               camp::tuple<SegmentTypes...> sharedSegments,
                               camp::idx_seq<Is...> seq) {
  //For the dimensions up to the piece number, we start at the shared start and go to the shared end
  auto startDims = tuple_slice<0,I>(sharedSegments);

  //For the dimension at the piece number, start at the shared end and go to the original end
  auto ithKnlSegment = camp::get<I>(knlSegments);
  auto ithSharedSegment = camp::get<I>(sharedSegments);
  auto ithDim = make_tuple(RangeSegment(*ithSharedSegment.end(), *ithKnlSegment.end()));

  //For the rest of the dimensions its start of shared to end of original
  auto endSeq = idx_seq_from_to<I+1, sizeof...(Is)>();
  auto endDims = high_boundary_end_dims(knlSegments, sharedSegments, endSeq);

  return tuple_cat(startDims, ithDim, endDims);
}

// Decomposes the upper boundary iteration space of a single kernel into a
// tuple of hyper-rectangles. It does so by using the upper faces of the sharedSegment
// as partition hyperplanes
template <typename...SegmentTypes, camp::idx_t...Is>
auto high_boundary_decomp(camp::tuple<SegmentTypes...> knlSegments,
                         camp::tuple<SegmentTypes...> sharedSegments,
                         camp::idx_seq<Is...> seq) {
  return make_tuple(high_boundary_decomp_piece<Is>(knlSegments, sharedSegments, seq)...);
}

// Returns the kernels that execute the upper boundary iterations for a single
// loop in a loop chain. The decomposition happens in the order of the iterations
// but their execution happens in the reverse order
template <typename KPol, typename SegmentTuple, typename... Bodies, typename...SegmentTypes, camp::idx_t...Is>
auto high_boundary_knls_for_knl(KernelWrapper<KPol,SegmentTuple,Bodies...> knl,
                                camp::tuple<SegmentTypes...> sharedSegmentTuple,
                                camp::idx_seq<Is...> seq) {
  auto highBoundaryIterSpaceDecomp = high_boundary_decomp(knl.segments, sharedSegmentTuple, seq);

  auto knls =  make_tuple(make_kernel<KPol>(camp::get<Is>(highBoundaryIterSpaceDecomp), camp::get<0>(knl.bodies))...);

  return tuple_reverse(knls);
}

// returns the kernels that execute the upper boundary iterations for a loopchain
template <typename...KernelTypes, camp::idx_t...Is>
auto high_boundary_knls(camp::tuple<KernelTypes...> knlTuple, camp::idx_seq<Is...>) {

  auto segmentTuples = make_tuple(camp::get<Is>(knlTuple).segments...);
  auto sharedIterSpaceTuple = intersect_segment_tuples(segmentTuples);

  return tuple_cat(high_boundary_knls_for_knl(camp::get<Is>(knlTuple),
                                            sharedIterSpaceTuple,
                                            idx_seq_for(sharedIterSpaceTuple))...);
}

template <camp::idx_t I, typename SegmentTuple>
RAJA_INLINE
auto low_boundary_start_dims_helper(SegmentTuple knlSegments, SegmentTuple sharedSegments) {
  auto knlSegment = camp::get<I>(knlSegments);
  auto sharedSegment = camp::get<I>(sharedSegments);

  using RangeType = decltype(knlSegment);
  return RangeType(*sharedSegment.begin(), *knlSegment.end());
}

template <typename SegmentTuple, camp::idx_t... Is>
RAJA_INLINE
auto low_boundary_start_dims(SegmentTuple knlSegments, SegmentTuple sharedSegments, camp::idx_seq<Is...>) {
  return make_tuple((low_boundary_start_dims_helper<Is>(knlSegments, sharedSegments))...);
}

// Creates one of the hyper-rectangles in the lower boundary
// iteration space decomposition for a single kernel
template <camp::idx_t I, typename...SegmentTypes, camp::idx_t...Is>
auto low_boundary_decomp_piece(camp::tuple<SegmentTypes...> knlSegments,
                               camp::tuple<SegmentTypes...> sharedSegments,
                               camp::idx_seq<Is...> seq) {
  //For the dimensions up to the piece number, we start at the shared start and go to the original end
  auto startSeq = idx_seq_from_to<0,I>();
  auto startDims = low_boundary_start_dims(knlSegments, sharedSegments, startSeq);

  //For the dimension at the piece number, start at the original start and go to the start of the shared
  auto ithKnlSegment = camp::get<I>(knlSegments);
  auto ithSharedSegment = camp::get<I>(sharedSegments);
  auto ithDim = make_tuple(RangeSegment(*ithKnlSegment.begin(), *ithSharedSegment.begin()));

  //For the rest of the dimensions its original.start to original.end
  auto endSeq = idx_seq_from_to<I+1, sizeof...(Is)>();
  auto endDims = tuple_slice<I+1, sizeof...(Is)>(knlSegments);

  return tuple_cat(startDims, ithDim, endDims);
}

// Decomposes the lower boundary iteration space of a single kernel into a
// tuple of hyper-rectangles. It does so by using the lower faces of the sharedSegment
// as partition hyperplanes
template <typename...SegmentTypes, camp::idx_t...Is>
auto low_boundary_decomp(camp::tuple<SegmentTypes...> knlSegments,
                         camp::tuple<SegmentTypes...> sharedSegments,
                         camp::idx_seq<Is...> seq) {
  return make_tuple(low_boundary_decomp_piece<Is>(knlSegments, sharedSegments, seq)...);
}


// Returns the kernels that execute the lower boundary iterations for a single
// loop in a loop chain.
template <typename KPol, typename SegmentTuple, typename... Bodies, typename...SegmentTypes, camp::idx_t...Is>
auto low_boundary_knls_for_knl(KernelWrapper<KPol,SegmentTuple,Bodies...> knl,
                               camp::tuple<SegmentTypes...> sharedSegmentTuple,
                               camp::idx_seq<Is...> seq) {
  auto lowBoundaryIterSpaceDecomp = low_boundary_decomp(knl.segments, sharedSegmentTuple, seq);

  return make_tuple(make_kernel<KPol>(camp::get<Is>(lowBoundaryIterSpaceDecomp), camp::get<0>(knl.bodies))...);
}


// returns the kernels that execute the lower boundary iterations for a loopchain
// Within the loop chain, there are iteration shared by all the loops in the chain.
// This function generates the kernels that execute the iterations that will be
// executed before the shared iterations
template <typename...KernelTypes, camp::idx_t...Is>
auto low_boundary_knls(camp::tuple<KernelTypes...> knlTuple, camp::idx_seq<Is...> seq) {

  auto segmentTuples = make_tuple(camp::get<Is>(knlTuple).segments...);
  auto sharedIterSpaceTuple = intersect_segment_tuples(segmentTuples);

  return tuple_cat(low_boundary_knls_for_knl(camp::get<Is>(knlTuple),
                                            sharedIterSpaceTuple,
                                            idx_seq_for(sharedIterSpaceTuple))...);


}

} //namespace RAJA
#endif
