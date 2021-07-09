// Contains the code for the transformation directives

#ifndef RAJA_LCTRANSFORMATIONS_HPP 
#define RAJA_LCTRANSFORMATIONS_HPP
#include "RAJA/config.hpp"

 
#include "RAJA/loopchain/Utils.hpp"
#include "RAJA/loopchain/LoopChain.hpp"
#include "RAJA/util/all-isl.h"
#include "RAJA/loopchain/KernelConversion.hpp"


namespace RAJA
{

template <typename... KernelTupleType>
struct LoopChain;

template <camp::idx_t MainKnlIdx, typename...KnlTupleType>
auto fused_kernels(camp::tuple<KnlTupleType...> knlTuple);

isl_union_set * fused_iterspace(isl_ctx * ctx, auto iterspaceTuple);

template <camp::idx_t... Dims>
auto fused_iterspace_low_bounds(isl_ctx * ctx, auto iterspaceTuple, camp::idx_seq<Dims...>);

template <camp::idx_t... Dims>
auto fused_iterspace_upper_bounds(isl_ctx * ctx, auto iterspaceTuple, camp::idx_seq<Dims...>);

template <typename...Knls>
auto chain(Knls...knls); 

/*

//returns a new segment that is segment shifted by shiftAmount
auto shift_segment(const auto segment, auto shiftAmount) {

  using RangeType = decltype(segment);

  auto low = *segment.begin();
  auto high = *segment.end();
 
  return RangeType(low+shiftAmount, high+shiftAmount);
  
} //shift_segment



//returns a new segment tuple where each segment is shifted by the pairwise amount in shiftAmountTuple
template <camp::idx_t...Is>
RAJA_INLINE
auto shift_segment_tuple(auto segmentTuple, auto shiftAmountTuple, camp::idx_seq<Is...> seq) {
  return make_tuple((shift_segment(camp::get<Is>(segmentTuple), camp::get<Is>(shiftAmountTuple)))...);
} // shift_segment_tuple



template <typename...Ts, camp::idx_t...Is>
auto shift_body(auto body, auto shiftAmountTuple, camp::idx_seq<Is...> seq) {

  auto newBody = [=](auto...args) {
    body((args - camp::get<Is>(shiftAmountTuple))...);
  }; 

  return newBody;


} //shift_body


template <typename KPol, typename SegmentTuple, typename...Bodies, camp::idx_t...Is>
RAJA_INLINE
auto shift(KernelWrapper<KPol,SegmentTuple,Bodies...> knl, auto shiftAmountTuple, camp::idx_seq<Is...> seq) 
{
  
  auto shiftedSegmentTuple = shift_segment_tuple(knl.segments, shiftAmountTuple, seq);

  auto shiftedBody = shift_body(camp::get<0>(knl.bodies), shiftAmountTuple, seq);

  return make_kernel<KPol>(shiftedSegmentTuple, shiftedBody);

} // shift


template <typename...Ts>
RAJA_INLINE
auto shift(auto knl, tuple<Ts...> shiftAmountTuple) 
{
  return shift(knl, shiftAmountTuple, idx_seq_for(shiftAmountTuple));
} // shift

template <typename...Ts>
auto shift(auto knl, Ts... shiftAmounts) {
  return shift(knl, make_tuple(shiftAmounts...)); 
} // shift

//
// Fusion Functions
//


template <camp::idx_t I>
RAJA_INLINE
auto pre_fuse_first_ith_dims_helper(auto originalSegments, auto overlapSegments) {

  auto fuseSegment = camp::get<I>(overlapSegments);
  auto originalSegment = camp::get<I>(originalSegments);

  using RangeType = decltype(fuseSegment);

  return RangeType(*fuseSegment.begin(), *originalSegment.end());

}

template <camp::idx_t... Is>
RAJA_INLINE
auto pre_fuse_first_ith_dims(auto originalSegments, auto overlapSegments, camp::idx_seq<Is...>) {
  
  return make_tuple((pre_fuse_first_ith_dims_helper<Is>(originalSegments, overlapSegments))...);  
}

template <camp::idx_t... Is>
RAJA_INLINE
auto pre_fuse_last_dims(auto originalSegments, camp::idx_seq<Is...>) {
  return make_tuple(camp::get<Is>(originalSegments)...);
}



//For each kernel of dimension d, there are d loops executed before the fused part.
//This function returns the segment tuple for the I-th one of these loops.
//Is... is a sequence from 0 to d-1. Its size is the number of dimensions in the loop
template <camp::idx_t I, camp::idx_t... Is>
RAJA_INLINE
auto pre_fuse_kernels_segments(auto originalSegments, auto overlapSegments, camp::idx_seq<Is...>) {
  
  //up to the I-th dimension, it is fuse.start to original.end

  auto firstIthSeq = idx_seq_from_to<0,I>();
  auto firstIthDims = pre_fuse_first_ith_dims(originalSegments, overlapSegments, firstIthSeq);
  
  // for the I-th, its orig.start to fuse.start

  auto ithOriginal = camp::get<I>(originalSegments);
  auto ithFuse = camp::get<I>(overlapSegments);

  using RangeType = decltype(ithOriginal);

  auto ithDim = make_tuple(RangeType(*ithOriginal.begin(), *ithFuse.begin()));

  // for ith to end, its original.start to original.end

  auto endSeq = idx_seq_from_to<I+1,sizeof...(Is)>();
  auto endDims = pre_fuse_last_dims(originalSegments, endSeq);

  return camp::tuple_cat_pair(firstIthDims, camp::tuple_cat_pair(ithDim, endDims));
  
  
  
}
template<camp::idx_t I>
RAJA_INLINE
auto post_fuse_last_dims_helper(auto originalSegments, auto overlapSegments) {
  
  auto fuseSegment = camp::get<I>(overlapSegments);
  auto originalSegment = camp::get<I>(originalSegments);

  using RangeType = decltype(fuseSegment);

  return RangeType(*fuseSegment.begin(), *originalSegment.end());


}

template <camp::idx_t...Is>
RAJA_INLINE
auto post_fuse_last_dims(auto originalSegments, auto overlapSegments, camp::idx_seq<Is...>) {

  
  return make_tuple((post_fuse_last_dims_helper<Is>(originalSegments, overlapSegments))...);  
}

//returns the segment tuple for the I-th post-fuse kernel
template <camp::idx_t I, camp::idx_t... Is>
RAJA_INLINE
auto post_fuse_kernels_segments(auto originalSegments, auto overlapSegments, camp::idx_seq<Is...>) {
  
  //up to the I-th dimension, it is fuse.start to fuse.end
  
  auto firstIthDims = slice_tuple<0,I>(overlapSegments);

  
  // for ith, it is fuse.end to original.end
  auto ithOriginal = camp::get<I>(originalSegments);
  auto ithFuse = camp::get<I>(overlapSegments);

  using RangeType = decltype(ithOriginal);

  auto ithDim = RangeType(*ithFuse.end(), *ithOriginal.end());

  // for ith to end, its fuse.start to original.end

  auto endSeq = idx_seq_from_to<I+1,sizeof...(Is)>();
  auto endDims = post_fuse_last_dims(originalSegments,overlapSegments, endSeq);

  
  auto postFuseSegments = tuple_cat(firstIthDims, make_tuple(ithDim), endDims);
  
  
  
  return postFuseSegments; 
}

template <camp::idx_t...Is>
RAJA_INLINE
auto zip_bounds(auto lows, auto highs, camp::idx_seq<Is...>) {

  return make_tuple((RangeSegment(camp::get<Is>(lows), camp::get<Is>(highs)+1))...);

}
template <camp::idx_t...Is>
auto fused_segment_tuple(auto knlTuple, camp::idx_seq<Is...>) {

  isl_ctx * ctx = isl_ctx_alloc();

  
  constexpr auto NumDims = camp::get<0>(knlTuple).numArgs;
  auto iterspaces = make_tuple(iterspace_from_knl<Is>(ctx, camp::get<Is>(knlTuple))...);
  
  auto lowPoint = fused_iterspace_low_bounds(ctx, iterspaces, camp::make_idx_seq_t<NumDims>{});
  auto highPoint = fused_iterspace_upper_bounds(ctx, iterspaces, camp::make_idx_seq_t<NumDims>{});

  return zip_bounds(lowPoint, highPoint, camp::make_idx_seq_t<NumDims>{});
 
}


//for the kernel with segmentTuple and a fused segment tuple fusedSegmentTuple, returns the tuple of segment tuples for the kernels that will execute the pre-fuse computation
template <camp::idx_t...Is>
RAJA_INLINE
auto pre_fuse_segment_tuples(auto segmentTuple, auto fusedSegmentTuple, camp::idx_seq<Is...> seq) {
  return make_tuple(pre_fuse_kernels_segments<Is>(segmentTuple, fusedSegmentTuple, seq)...);
}
auto pre_fuse_segment_tuples(auto segmentTuple, auto fusedSegmentTuple) {
  return pre_fuse_segment_tuples(segmentTuple, fusedSegmentTuple, idx_seq_for(segmentTuple));
}


//returns the pre-fuse kernels that are responsible for the pre-fuse iteration section of one kernel in the knlTuple
template <typename KPol, typename SegmentTuple, typename...Bodies, camp::idx_t...Is>
RAJA_INLINE
auto pre_fuse_knls_from_knl(KernelWrapper<KPol,SegmentTuple,Bodies...> knl, auto fuseSegmentTuple, camp::idx_seq<Is...>) {
  
   auto preFuseSegmentTuples = pre_fuse_segment_tuples(knl.segments, fuseSegmentTuple);

   return make_tuple((make_kernel<KPol>(camp::get<Is>(preFuseSegmentTuples), camp::get<0>(knl.bodies)))...);

}//pre_fuse_knls_from_knl

template <typename KernelPol, typename Segments, typename...Bodies>
RAJA_INLINE
auto pre_fuse_knls_from_knl(KernelWrapper<KernelPol,Segments,Bodies...> knl, auto fusedSegmentTuple) {
  return pre_fuse_knls_from_knl(knl, fusedSegmentTuple, camp::make_idx_seq_t<knl.numArgs>{});
} //pre_fuse_knls_from_knl


template <typename...KnlTupleTypes, camp::idx_t...Is>
RAJA_INLINE
auto pre_fuse_knls(auto knlTuple, camp::idx_seq<Is...> seq) {

  //std::cout << "This function needs to concat all the kernels created from each knl in the tuple\n";
  auto fusedSegmentTuple = fused_segment_tuple(knlTuple, seq);
  return tuple_cat((pre_fuse_knls_from_knl(camp::get<Is>(knlTuple), fusedSegmentTuple))...); 
}

//given the segment tuple for an entire loop and the segment tuple for the fused area of that loop,
//creates the segment tuples for the kernels that execute the parts of the kernel that are executed after the fused area
template <camp::idx_t...Is>
RAJA_INLINE
auto post_fuse_segment_tuples(auto segmentTuple, auto fusedSegmentTuple, camp::idx_seq<Is...> seq) {
  return make_tuple(post_fuse_kernels_segments<Is>(segmentTuple, fusedSegmentTuple, seq)...);
}
auto post_fuse_segment_tuples(auto segmentTuple, auto fusedSegmentTuple) {
  return post_fuse_segment_tuples(segmentTuple, fusedSegmentTuple, idx_seq_for(segmentTuple));
}

template <typename KPol, typename SegmentTuple, typename...Bodies, camp::idx_t...Is>
RAJA_INLINE
auto post_fuse_knls_from_knl(KernelWrapper<KPol,SegmentTuple,Bodies...> knl, auto fuseSegmentTuple, camp::idx_seq<Is...>) {
   
   auto postFuseSegmentTuples = post_fuse_segment_tuples(knl.segments, fuseSegmentTuple);

   return make_tuple((make_kernel<KPol>(camp::get<Is>(postFuseSegmentTuples), camp::get<0>(knl.bodies)))...);
}//post_fuse_knls_from_knl

template <typename KernelPol, typename Segments, typename...Bodies>
RAJA_INLINE
auto post_fuse_knls_from_knl(KernelWrapper<KernelPol,Segments,Bodies...> knl, auto fusedSegmentTuple) {
  return post_fuse_knls_from_knl(knl, fusedSegmentTuple, camp::make_idx_seq_t<knl.numArgs>{});
} //post_fuse_knls_from_knl



template <typename...KnlTupleTypes, camp::idx_t...Is>
RAJA_INLINE
auto post_fuse_knls(auto knlTuple, camp::idx_seq<Is...> seq) {
  
  auto fusedSegmentTuple = fused_segment_tuple(knlTuple, seq);
  return tuple_cat((post_fuse_knls_from_knl(camp::get<Is>(knlTuple), fusedSegmentTuple))...);
}


template <camp::idx_t... Is>
RAJA_INLINE
auto fused_lambda(auto bodies, camp::idx_seq<Is...>) {
  return [=](auto...is) {
    camp::sink((camp::get<Is>(bodies)(is...), 0)...);
  };

}

template <camp::idx_t I1, camp::idx_t I2, camp::idx_t I3,
          camp::idx_t I4, camp::idx_t I5, camp::idx_t I6, 
          camp::idx_t I7, camp::idx_t I8, camp::idx_t I9, 
          camp::idx_t I10, camp::idx_t I11>
RAJA_INLINE
auto fused_lambda(auto bodies, camp::idx_seq<I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11>) {
  return [=](auto...is) {
    camp::get<0>(bodies)(is...);
    camp::get<1>(bodies)(is...);
    camp::get<2>(bodies)(is...);
    camp::get<3>(bodies)(is...);
    camp::get<4>(bodies)(is...);
    camp::get<5>(bodies)(is...);
    camp::get<6>(bodies)(is...);
    camp::get<7>(bodies)(is...);
    camp::get<8>(bodies)(is...);
    camp::get<9>(bodies)(is...);
    camp::get<10>(bodies)(is...);

  };
}


template <camp::idx_t I1, camp::idx_t I2, camp::idx_t I3,
          camp::idx_t I4, camp::idx_t I5, camp::idx_t I6>
RAJA_INLINE
auto fused_lambda(auto bodies, camp::idx_seq<I1,I2,I3,I4,I5,I6>) {
  return [=](auto...is) {
    camp::get<0>(bodies)(is...);
    camp::get<1>(bodies)(is...);
    camp::get<2>(bodies)(is...);
    camp::get<3>(bodies)(is...);
    camp::get<4>(bodies)(is...);
    camp::get<5>(bodies)(is...);
  };
}

template <camp::idx_t I1, camp::idx_t I2, camp::idx_t I3>
RAJA_INLINE
auto fused_lambda(auto bodies, camp::idx_seq<I1,I2,I3>) {
  return [=](auto...is) {
    camp::get<0>(bodies)(is...);
    camp::get<1>(bodies)(is...);
    camp::get<2>(bodies)(is...);
  };
}
template <camp::idx_t I1, camp::idx_t I2>
RAJA_INLINE
auto fused_lambda(auto bodies, camp::idx_seq<I1,I2> seq) {
  return [=](auto...is) {
    camp::get<0>(bodies)(is...);
    camp::get<1>(bodies)(is...);
  };
}


template <camp::idx_t I>
RAJA_INLINE
auto fused_lambda(auto bodies, camp::idx_seq<I> seq) {
  return [=](auto...is) {
    camp::get<0>(bodies)(is...);
  };
}


template <typename...KnlTupleTypes, camp::idx_t...Is>
RAJA_INLINE
auto fused_knl(auto knlTuple, camp::idx_seq<Is...> seq) {
  auto fusedSegmentTuple = fused_segment_tuple(knlTuple, seq);
  
  auto lambdas = make_tuple((camp::get<0>(camp::get<Is>(knlTuple).bodies))...); 
  auto fusedLambda = fused_lambda(lambdas, seq);
  
  auto knl = camp::get<0>(knlTuple);
  using KPol = typename decltype(knl)::KPol;
  return make_kernel<KPol>(fusedSegmentTuple, fusedLambda);
}


template <typename...KnlTupleTypes, camp::idx_t...Is>
RAJA_INLINE
auto fuse(camp::tuple<KnlTupleTypes...> knlTuple, camp::idx_seq<Is...> seq) {
  
  //std::cout << "starting fuse\n" << std::flush; 
  auto preKnls = pre_fuse_knls(knlTuple, seq);
  auto postKnls = post_fuse_knls(knlTuple, seq);
  auto fusedKnl = fused_knl(knlTuple, seq);

  auto allFusedKnls = tuple_cat(preKnls, make_tuple(fusedKnl), postKnls);

  using preKnlType = decltype(preKnls);

  constexpr camp::idx_t knlIdx = camp::tuple_size<preKnlType>::value;
  //std::cout << "done with fuse\n" << std::flush;
  return fused_kernels<knlIdx>(allFusedKnls);
  

} //fuse


template < typename...KnlTupleTypes>
RAJA_INLINE
auto fuse(camp::tuple<KnlTupleTypes...> knlTuple) {
  return fuse(knlTuple, idx_seq_for(knlTuple));
}

template <typename...Knls>
RAJA_INLINE
auto fuse(Knls...knls) {
  return fuse(make_tuple(knls...), idx_seq_for(make_tuple(knls...)));
}

*/
//
// Shift and Fuse Functions
//

/*
//returns the constraints on the shift values created from the dependences between the knls
std::string shift_constraint(auto knl1, auto knl2, auto id1, auto id2 ) {
  
  constexpr int numDims = knl1.numArgs;

  isl_ctx * ctx = isl_ctx_alloc();
  //isl_printer * p = isl_printer_to_file(ctx, stdout);

  isl_union_map * depRelation = data_dep_relation_from_knls<0,1>(ctx, knl1,knl2);

  isl_union_set * ispace1 = iterspace_from_knl<0>(ctx, knl1);

  auto maxInput = isl_union_set_sample_point(isl_union_set_lexmax(ispace1));
  auto image = isl_union_set_apply(isl_union_set_from_point(isl_point_copy(maxInput)), depRelation);
  
  if(isl_union_set_is_empty(image)) {
    return "";
  }
  //now we want to get the minimum value in each dimension of the image elements;
  
  

  int * minValues = new int[numDims+1];
  minValues[0] = numDims;
  for(int i = 1; i <= numDims; i++) { minValues[i] = std::numeric_limits<int>::max(); }

  int * inputPointValues = new int[numDims+1];
  inputPointValues[0] = numDims;
  for(int i = 1; i <= numDims; i++) {inputPointValues[i] = std::numeric_limits<int>::max(); }

  auto update_min_values = [] (isl_point * p, void * currMin_) {
    int * currMin = (int *) (currMin_);
    int numDims = currMin[0];
    for(int i = 0; i < numDims; i++) {
      isl_val * pointVal = isl_point_get_coordinate_val(p, isl_dim_set, i);
      int value = isl_val_get_num_si(pointVal);
      //std::cout << "point value: " << value << "\n";
      if(currMin[i+1] > value) {
        currMin[i+1] = value;
        //std::cout << "changed\n";
      }
    }
    return isl_stat_ok;
  }; //update_min_values

  //collect the point values for the input point
  update_min_values(maxInput, (void*) inputPointValues);
  //collect the minimum values for the output points
  isl_union_set_foreach_point(image, update_min_values, (void*) minValues);
  for(int i = 0; i < numDims; i++) {
   // std::cout << "dim " << i << " input: " << inputPointValues[i+1] << "\n";
   // std::cout << "output minimum " << i << " is " << minValues[i+1]<< "\n";;
  }

  //Now that we have the output minumums for the input point, we need to set up the constraint.
  //For each dimension, we have input - outputMax = shift2 - shift1 
  std::string constraints = "";
  for(int i = 0; i < numDims; i++) {
    std::string constraint = "";
    constraint += std::to_string(inputPointValues[i+1]) + "-" + std::to_string(minValues[i+1]) + "=";
    constraint += "S" + std::to_string(id2) + "_" + std::to_string(i);
    constraint += " - ";
    constraint += "S" + std::to_string(id1) + "_" + std::to_string(i);
    //std::cout << "Constraint " << i << ": " << constraint << "\n";;
   constraints += constraint;
    if (i != numDims - 1) {constraints += " and ";}
  }
  //std::cout << "Constraints: " << constraints << "\n"; 

  return constraints;
  

}

template <camp::idx_t KnlId1, camp::idx_t KnlId2, camp::idx_t NumKnls>
auto shift_constraints_helper(isl_ctx * ctx, auto knlTuple) {

  if constexpr (KnlId1 > NumKnls) {
    std::string terminal = "END";
    return terminal;
  } else if constexpr (KnlId2 > NumKnls) {
    return shift_constraints_helper<KnlId1+1, KnlId1+2, NumKnls>(ctx, knlTuple);
  } else {
    auto constraintString = shift_constraint(camp::get<KnlId1-1>(knlTuple), camp::get<KnlId2-1>(knlTuple), KnlId1, KnlId2);
    auto rest = shift_constraints_helper<KnlId1, KnlId2+1, NumKnls>(ctx, knlTuple);

    if(rest == "END") {
      return constraintString;
    } else if (constraintString == "") {
      return rest;
    } else {
      return constraintString + " and " + rest;
    }
  }
 

}

template <camp::idx_t...Is>
auto shift_constraints(isl_ctx * ctx, auto knlTuple, camp::idx_seq<Is...> knlSeq) {
  return shift_constraints_helper<1,2,sizeof...(Is)>(ctx, knlTuple);
} //shift_constraint_sets


template <camp::idx_t NumKnls, camp::idx_t...Dims>
auto shift_vector_to_shift_tuples(std::vector<int> shiftVector, camp::idx_seq<Dims...> dimSeq) {
  if constexpr (NumKnls == 0) {
    return make_tuple();
  } else {
    auto currTuple = make_tuple(shiftVector.at(Dims)...);
    std::vector<int> remainingShifts = {};
    for(int i = sizeof...(Dims); i < shiftVector.size(); i++) {
      remainingShifts.push_back(shiftVector.at(i));
    }
    auto restTuple = shift_vector_to_shift_tuples<NumKnls-1>(remainingShifts, dimSeq);
 
    return tuple_cat(make_tuple(currTuple), restTuple);
  }
}


template <camp::idx_t NumKnls, camp::idx_t NumDims>
auto shift_vector_to_shift_tuples(std::vector<int> shiftVector) {
  return shift_vector_to_shift_tuples<NumKnls>(shiftVector, camp::make_idx_seq_t<NumDims>{});
}

//returns a tuple of shift amount tuples for each of the kernels in the knlTuple
template <camp::idx_t...Is>
auto shift_amount_tuples(auto knlTuple, camp::idx_seq<Is...> seq) {
  constexpr int numKernels = sizeof...(Is);
  constexpr int numDims = camp::get<0>(knlTuple).numArgs;

  isl_ctx * ctx = isl_ctx_alloc();
  isl_printer * p = isl_printer_to_file(ctx, stdout);
  auto deprel = dependence_relation_from_kernels(ctx, knlTuple, seq);

  
  //std::cout << "Dependence relation among kernels\n";
  //p= isl_printer_print_union_map(p,deprel);
  //std::cout << "\n";

  //First, we create the string for the shift variables and the constraint that they are all greater than = 0.
  std::string shiftSet = "[";
  std::string nonNegConstraint = "";
  for(int loopNum = 1; loopNum <= numKernels; loopNum++) {
    for(int dim = 0; dim < numDims; dim++) {
      std::string shiftString = "S" + std::to_string(loopNum) + "_" + std::to_string(dim);
      
      shiftSet += shiftString;
      nonNegConstraint += shiftString + " >= 0 ";
    
      if(dim != numDims - 1) {
       shiftSet += ",";
       nonNegConstraint += " and ";
      }
    }//dim
    if(loopNum != numKernels) {
      shiftSet += ",";
      nonNegConstraint += " and ";
    }
  }//loopNum

  shiftSet += "]";
e
  //std::cout << "Shift Set String: " << shiftSet << "\n";
  //std::cout << "Non Negative Constraint: " << nonNegConstraint << "\n";

  std::string shiftSpaceString = "{" + shiftSet + ": " + nonNegConstraint + "}";
  isl_set * shiftSpace = isl_set_read_from_str(ctx, shiftSpaceString.c_str());

  //std::cout << "Shift space: ";
  //p = isl_printer_print_set(p, shiftSpace);
  //std::cout << "\n";

  auto constraints = shift_constraints(ctx, knlTuple, seq);
  
  //std::cout << "Entire Constraint String\n";
  //std::cout << constraints << "\n\n"; 

  std::string constrainedShiftSetString = "{" + shiftSet + " : " + nonNegConstraint + " and " + constraints + "}";

  isl_set * possibleShifts = isl_set_read_from_str(ctx, constrainedShiftSetString.c_str());

  //std::cout << "Set of all legalizing shifts:\n";
  //p = isl_printer_print_set(p, possibleShifts);
  //std::cout << "\n";

  isl_point * smallestShifts = isl_set_sample_point(isl_set_lexmin(possibleShifts));
  
  auto point_to_vector = [](isl_point * p, int numDims) {
    std::vector<int> vals = {};

    for(int i = 0; i < numDims; i++) {
      isl_val * pointVal = isl_point_get_coordinate_val(p, isl_dim_set, i);
      int value = isl_val_get_num_si(pointVal);
      vals.push_back(value);
    }
    return vals;
  };

  auto shiftAmounts = point_to_vector(smallestShifts, numKernels * numDims);
 
  return shift_vector_to_shift_tuples<numKernels,numDims>(shiftAmounts);
}//shift_amount_tuples

template <camp::idx_t...Is>
auto shift_and_fuse(auto knlTuple, camp::idx_seq<Is...> seq) {
  auto shiftAmountTuples = shift_amount_tuples(knlTuple, seq);
  auto shiftedKnls = make_tuple(shift(camp::get<Is>(knlTuple), camp::get<Is>(shiftAmountTuples))...);
  return fuse(shiftedKnls); 
}//shift_and_fuse

template <typename...Knls>
auto shift_and_fuse(Knls...knls) {
  return shift_and_fuse(make_tuple(knls...), idx_seq_for(make_tuple(knls...))); 
}//shift_and_fuse
*/
//
// Overlapped Tile Without Fusion Functions
//
/*

//creates the constraints for the overlap amounts between two kernels
//the overlap amount constraint is that for a particular dimension, 
//the overlap amonut for knl1 has to be at least the amount for knl2
// plus the maximum length of the dependences between the kernels in
// that dimension. knl1_dim >= knl2_dim + maxDep(knl1, knl2, dim)
//
std::string overlap_constraint(auto knl1, auto knl2, auto id1, auto id2) {
  
  //std::cerr << "Calculating the overlap constraints between kernels " << id1 << " and " << id2 << "\n";
  constexpr int numDims = knl1.numArgs;

  isl_ctx * ctx = isl_ctx_alloc();
  //isl_printer * p = isl_printer_to_file(ctx, stdout);

  isl_union_map * depRelation = data_dep_relation_from_knls<0,1>(ctx, knl1,knl2);

  //isl_union_set * ispace1 = iterspace_from_knl<0>(ctx, knl1);

  isl_union_set * minInput = isl_union_set_lexmin(iterspace_from_knl<0>(ctx, knl1));

  isl_union_set * image = isl_union_set_apply(isl_union_set_copy(minInput), depRelation);
  if(isl_union_set_is_empty(image)) {
    //std::cerr << "No interaction between kernels\n";
    return "";
  }

  int * maxOut = new int[numDims + 1];
  int * inputValues = new int[numDims + 1];
  
  maxOut[0] = numDims;
  inputValues[0] = numDims;
 
  auto update_max_values = [] (isl_point * p, void * currMax_) {
    int * currMax = (int *) (currMax_);
    int numDims = currMax[0];
    for(int i = 0; i < numDims; i++) {
      isl_val * pointVal = isl_point_get_coordinate_val(p, isl_dim_set, i);
      int value = isl_val_get_num_si(pointVal);
      if(currMax[i+1] < value) {
        currMax[i+1] = value;
      }
    }
    return isl_stat_ok;
  }; //update_max_values
 
  for(int i = 1; i <= numDims; i++) {
    maxOut[i] = std::numeric_limits<int>::min();
    inputValues[i] = std::numeric_limits<int>::min();
  }

  //set the inputValues array to the point used
  update_max_values(isl_union_set_sample_point(minInput), inputValues);

  for(int i = 0; i <= numDims; i++) {
    //std::cout << "minInput[" << i << "] = " << inputValues[i] << "\n";
  }
  isl_union_set_foreach_point(isl_union_set_copy(image), update_max_values, maxOut);

  for(int i = 0; i <= numDims; i++) {
    //std::cout << "maxOut[" << i << "] = " << maxOut[i] << "\n";
  }

  auto maxDependence = new int[numDims+1];
  for(int i = 0; i <= numDims; i++) {
    maxDependence[i] = maxOut[i] - inputValues[i];
    //std::cout << "maxDependence[" << i << "] = " << maxDependence[i] << "\n";
  }

  std::string constraints = "";
  for(int i = 0; i < numDims; i++) {
    std::string constraint = "";
    constraint += "O" + std::to_string(id1) + "_" + std::to_string(i);
    constraint += " >= ";
    constraint += "O" + std::to_string(id2) + "_" + std::to_string(i);
    constraint += "+"; 
    constraint += std::to_string(maxDependence[i+1]);
    //std::cout << "Constraint " << i << ": " << constraint << "\n";;
    constraints += constraint;
    if (i != numDims - 1) {constraints += " and ";}
  }
  //std::cout << "Constraints: " << constraints << "\n"; 

  return constraints;

} //overlap_amount_constraint

template <camp::idx_t KnlId1, camp::idx_t KnlId2, camp::idx_t NumKnls>
auto overlap_constraints_helper(isl_ctx * ctx, auto knlTuple) {

  if constexpr (KnlId1 > NumKnls) {
    std::string terminal = "END";
    return terminal;
  } else if constexpr (KnlId2 > NumKnls) {
    return overlap_constraints_helper<KnlId1+1, KnlId1+2, NumKnls>(ctx, knlTuple);
  } else {
    auto constraintString = overlap_constraint(camp::get<KnlId1-1>(knlTuple), camp::get<KnlId2-1>(knlTuple), KnlId1, KnlId2);
    auto rest = overlap_constraints_helper<KnlId1, KnlId2+1, NumKnls>(ctx, knlTuple);

    if(rest == "END") {
      return constraintString;
    } else if (constraintString == "") {
      return rest;
    } else {
      return constraintString + " and " + rest;
    }
  }
 

}

template <camp::idx_t...Is>
auto overlap_constraints(isl_ctx * ctx, auto knlTuple, camp::idx_seq<Is...> knlSeq) {
  return overlap_constraints_helper<1,2,sizeof...(Is)>(ctx, knlTuple);
} //shift_constraint_sets



template <camp::idx_t...Is>
auto overlap_amount_tuples(auto knlTuple, camp::idx_seq<Is...> seq) {
  constexpr auto numKernels = sizeof...(Is);
  auto constexpr numDims = camp::get<0>(knlTuple).numArgs;
 
  //std::cout << "calculating overlap amounts\n"; 
  //pairwise maximum dependence distance as constraints.
  isl_ctx * ctx = isl_ctx_alloc();

  std::string overlapSet = "[";
  std::string nonNegConstraint = "";
  for(int loopNum = 1; loopNum <= numKernels; loopNum++) {
    for(int dim = 0; dim < numDims; dim++) {
      std::string overlapString = "O" + std::to_string(loopNum) + "_" + std::to_string(dim);
      overlapSet += overlapString;

      nonNegConstraint += overlapString + " >= 0 ";
    
      if(dim != numDims - 1) {
       overlapSet += ",";
       nonNegConstraint += " and ";
      }
    }//dim
    if(loopNum != numKernels) {
      overlapSet += ",";
      nonNegConstraint += " and ";
    }
  }//loopNum
  
  overlapSet += "]";

  std::string legalOverlapsString = "{" + overlapSet + ": " + nonNegConstraint + " and " + overlap_constraints(ctx, knlTuple, seq) + "}";

  isl_union_set * legalOverlaps = isl_union_set_read_from_str(ctx, legalOverlapsString.c_str());

  isl_point * legalOverlap = isl_union_set_sample_point(isl_union_set_lexmin(legalOverlaps));
  
  auto point_to_vector = [](isl_point * p, int numDims) {
    std::vector<int> vals = {};

    for(int i = 0; i < numDims; i++) {
      isl_val * pointVal = isl_point_get_coordinate_val(p, isl_dim_set, i);
      int value = isl_val_get_num_si(pointVal);
      vals.push_back(value);
    }
    return vals;
  };

  auto overlapAmounts = point_to_vector(legalOverlap, numKernels * numDims);
  return shift_vector_to_shift_tuples<numKernels,numDims>(overlapAmounts);

} //overlap_amount_tuples
//adds the overlap amonut to the begining of the segment
auto add_overlap(auto segment, auto originalSegment, std::size_t overlapAmount) {

  auto beg = *(segment.begin());
  auto end = *(segment.end());
  signed long int newBeg = beg - overlapAmount;
 
  //std::cout << "Adding overlap of " << overlapAmount << " to the range " << beg << "," << end << "getting " << newBeg << "," << end << "\n";

  auto originalBeg = *(originalSegment.begin());

  if(newBeg < originalBeg) {
    return RangeSegment(originalBeg, end);
  } else {
    return RangeSegment(newBeg, end);
  }
}

template <camp::idx_t...Is>
auto overlap_segment_tuple(auto segmentTuple, auto originalSegmentTuple, auto overlap, camp::idx_seq<Is...>) {

  return make_tuple((add_overlap(camp::get<Is>(segmentTuple), camp::get<Is>(originalSegmentTuple), camp::get<Is>(overlap)))...);


}
auto overlap_segment_tuple(auto segmentTuple, auto originalSegmentTuple, auto overlap ) {
  return overlap_segment_tuple(segmentTuple, originalSegmentTuple, overlap, idx_seq_for(overlap));
}
//creates a overlapped tiling kernel policy for loops with NumDims dimensions
template <camp::idx_t TileSize, camp::idx_t CurrDim, camp::idx_t NumDims>
auto overlapped_tile_policy() {
  if constexpr (CurrDim == NumDims) {
    return statement::TiledLambda<0>{};
  } else {
    auto subPolicy = overlapped_tile_policy<TileSize, CurrDim+1, NumDims>();
    using subPolicyType = decltype(subPolicy);
    return statement::OverlappedTile<CurrDim, statement::tile_fixed<TileSize>, RAJA::omp_parallel_for_exec, subPolicyType>{};
  }
}

//returns the lambda that executes a particular tile with overlaps
template <camp::idx_t...Is>
auto overlapped_tile_no_fuse_executor(auto knlTuple, auto overlaps, camp::idx_seq<Is...> seq) {

  //std::cout << "overlap executor no fuse for " << sizeof...(Is) << " loops of " << tuple_len(camp::get<0>(overlaps)) << "dims\n";

  auto entireRangeSegment = fused_segment_tuple(knlTuple, seq);

  auto f = [=](auto tileRangeSegments) {
    
     auto overlapTileSegmentTuples = make_tuple((overlap_segment_tuple(tileRangeSegments, entireRangeSegment, camp::get<Is>(overlaps)))...);
     auto newKnlTuple = make_tuple((change_segment_tuple(camp::get<Is>(knlTuple), camp::get<Is>(overlapTileSegmentTuples)))...);
     auto overlappedTileChain = chain(camp::get<Is>(newKnlTuple)...);
     overlappedTileChain();
  };
  
  return f;
}

auto overlapped_tile_no_fuse_executor(auto knlTuple, auto overlaps) {
  return overlapped_tile_no_fuse_executor(knlTuple, overlaps, idx_seq_for(knlTuple)); 
} //overlap_executor


//creates the kernel which tiles the overlapping region of the kernels in the tuple
template <camp::idx_t TileSize, camp::idx_t...Is>
auto overlap_tile_no_fuse_kernel(auto knlTuple, auto overlapAmountTuples, camp::idx_seq<Is...> knlSeq) {
  
  auto constexpr numDims = camp::get<0>(knlTuple).numArgs;
  using InnerPol = decltype(overlapped_tile_policy<TileSize,0,numDims>());
  using KPol = KernelPolicy<InnerPol>;

  auto segmentTuple = fused_segment_tuple(knlTuple, knlSeq);
  auto tiledExecutor = overlapped_tile_no_fuse_executor(knlTuple, overlapAmountTuples);

  return make_kernel<KPol>(segmentTuple, tiledExecutor);
}//overlap_kernel



template <camp::idx_t TileSize, camp::idx_t...Is>
auto overlapped_tile_no_fuse(auto knlTuple, camp::idx_seq<Is...> seq) {
  
  auto shiftAmountTuples = shift_amount_tuples(knlTuple, seq);

  auto shiftedKnlTuple = make_tuple(shift(camp::get<Is>(knlTuple), camp::get<Is>(shiftAmountTuples))...);

  auto preTilingKnls = pre_fuse_knls(shiftedKnlTuple, seq);
  auto postTilingKnls = post_fuse_knls(shiftedKnlTuple, seq);

  
  auto overlapAmountTuples = overlap_amount_tuples(shiftedKnlTuple, seq);

  auto overlappedKernel = overlap_tile_no_fuse_kernel<TileSize, Is...>(shiftedKnlTuple, overlapAmountTuples, seq);

  auto allKernelTuple = tuple_cat(preTilingKnls, make_tuple(overlappedKernel), postTilingKnls);

  return fused_kernels<1>(allKernelTuple);
}//overlapped_tile_no_fuse



template <camp::idx_t TileSize, typename...Knls>
auto overlapped_tile_no_fuse(Knls...knls) {
  auto knlTuple = make_tuple(knls...);
  auto seq = idx_seq_for(knlTuple);
  
  return overlapped_tile_no_fuse<TileSize>(knlTuple, idx_seq_for(knlTuple));
}


template <typename...Knls>
auto overlapped_tile_no_fuse(Knls...knls) {
  constexpr auto TileSize = 16;
  auto knlTuple = make_tuple(knls...);
  return overlapped_tile_no_fuse<TileSize>(knlTuple, idx_seq_for(knlTuple));
} //overlapped_tile_no_fuse




//returns the lambda that executes a particular tile with overlaps
template <camp::idx_t...Is>
auto overlapped_tile_fuse_executor(auto knlTuple, auto overlaps, camp::idx_seq<Is...> seq) {

  //std::cout << "overlap executor fuse for " << sizeof...(Is) << " loops of " << tuple_len(camp::get<0>(overlaps)) << "dims\n";

  auto entireRangeSegment = fused_segment_tuple(knlTuple, seq);

  auto f = [=](auto tileRangeSegments) {
    
     auto overlapTileSegmentTuples = make_tuple((overlap_segment_tuple(tileRangeSegments, entireRangeSegment, camp::get<Is>(overlaps)))...);
     auto newKnlTuple = make_tuple((change_segment_tuple(camp::get<Is>(knlTuple), camp::get<Is>(overlapTileSegmentTuples)))...);
     auto overlappedTileChain = fuse(camp::get<Is>(newKnlTuple)...);
     overlappedTileChain();
  };
  
  return f;
}

auto overlapped_tile_fuse_executor(auto knlTuple, auto overlaps) {
  return overlapped_tile_fuse_executor(knlTuple, overlaps, idx_seq_for(knlTuple)); 
} //overlap_executor


//creates the kernel which tiles the overlapping region of the kernels in the tuple
template <camp::idx_t...Is>
auto overlap_tile_fuse_kernel(auto knlTuple, auto overlapAmountTuples, camp::idx_seq<Is...> knlSeq) {
  
  auto constexpr numDims = camp::get<0>(knlTuple).numArgs;
  using InnerPol = decltype(overlapped_tile_policy<128,0,numDims>());
  using KPol = KernelPolicy<InnerPol>;

  auto segmentTuple = fused_segment_tuple(knlTuple, knlSeq);
  auto tiledExecutor = overlapped_tile_fuse_executor(knlTuple, overlapAmountTuples);

  return make_kernel<KPol>(segmentTuple, tiledExecutor);
}//overlap_kernel
template <camp::idx_t...Is>
auto overlapped_tile_fuse(auto knlTuple, camp::idx_seq<Is...> seq) {
  
  auto shiftAmountTuples = shift_amount_tuples(knlTuple, seq);

  auto shiftedKnlTuple = make_tuple(shift(camp::get<Is>(knlTuple), camp::get<Is>(shiftAmountTuples))...);

  auto preTilingKnls = pre_fuse_knls(shiftedKnlTuple, seq);
  auto postTilingKnls = post_fuse_knls(shiftedKnlTuple, seq);

  
  auto overlapAmountTuples = overlap_amount_tuples(shiftedKnlTuple, seq);

  auto overlappedKernel = overlap_tile_fuse_kernel(shiftedKnlTuple, overlapAmountTuples, seq);

  auto allKernelTuple = tuple_cat(preTilingKnls, make_tuple(overlappedKernel), postTilingKnls);

  return fused_kernels<1>(allKernelTuple);
}//overlapped_tile_fuse

template <typename...Knls>
auto overlapped_tile_fuse(Knls...knls) {
  auto knlTuple = make_tuple(knls...);
  return overlapped_tile_fuse(knlTuple, idx_seq_for(knlTuple));

}
*/
template <typename...Knls>
auto chain(Knls...knls) {
  return LoopChain<Knls...>(make_tuple(knls...));
}



} //namespace RAJA

#endif
