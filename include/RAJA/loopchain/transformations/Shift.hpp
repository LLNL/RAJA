// Contains code for shift transformation

#ifndef RAJA_LCSHIFT_HPP
#define RAJA_LCSHIFT_HPP

#include "RAJA/config.hpp"
#include "RAJA/loopchain/KernelWrapper.hpp"

namespace RAJA 
{

//Transformation Declarations

template <typename KernelType, typename...Amounts>
auto shift(KernelType knl, camp::tuple<Amounts...> shiftAmountTuple);

template <typename KernelType, typename...Amounts>
auto shift(KernelType knl, Amounts...shiftAmounts);

//Transformation Definitions

// shifts a single range segment by shiftAmount
template <typename ShiftAmountType>
auto shift_segment(RangeSegment segment, const ShiftAmountType shiftAmount) {
  auto low = *segment.begin();
  auto high = *segment.end();

  return RangeSegment(low+shiftAmount, high+shiftAmount);
}

// shifts the segments in segmentTuple by the amounts in shiftAmountTuple
template <typename SegmentTuple, typename ShiftAmountTuple, camp::idx_t...Is>
auto shift_segment_tuple(SegmentTuple segmentTuple, ShiftAmountTuple shiftAmountTuple, camp::idx_seq<Is...>) {
  return make_tuple((shift_segment(camp::get<Is>(segmentTuple), camp::get<Is>(shiftAmountTuple)))...);
}


// shifts the arguments to a lambda backwards by the amonuts in shiftAmountTuple
template <typename Body, typename...Amounts, camp::idx_t...Is>
auto shift_body(Body body, camp::tuple<Amounts...> shiftAmountTuple, camp::idx_seq<Is...>) {
  auto newBody = [=](auto...args) {
    body((args - camp::get<Is>(shiftAmountTuple))...);
  };
  return newBody;
}


template <typename KernelType, typename...Amounts, camp::idx_t...Is>
auto shift(KernelType knl, camp::tuple<Amounts...> shiftAmountTuple, camp::idx_seq<Is...> seq) {
  auto shiftedSegmentTuple = shift_segment_tuple(knl.segments, shiftAmountTuple, seq);
  auto shiftedBody = shift_body(camp::get<0>(knl.bodies), shiftAmountTuple, seq);
  
  using KPol = typename KernelType::KPol;
  return make_kernel<KPol>(shiftedSegmentTuple, shiftedBody);
}


template <typename KernelType, typename...Amounts>
auto shift(KernelType knl, camp::tuple<Amounts...> shiftAmountTuple) {
  return shift(knl, shiftAmountTuple, idx_seq_for(shiftAmountTuple));
}

template <typename KernelType, typename...Amounts>
auto shift(KernelType knl, Amounts...shiftAmounts) {
  auto amountTuple = camp::make_tuple(shiftAmounts...);
  return shift(knl, amountTuple, idx_seq_for(amountTuple));
}




} //namespace RAJA


#endif
