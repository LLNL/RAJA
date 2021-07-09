
#ifndef RAJA_ISL_ANALYSIS_HPP
#define RAJA_ISL_ANALYSIS_HPP

#include "RAJA/loopchain/LoopChain.hpp"
#include "RAJA/loopchain/KernelConversion.hpp"

#include "RAJA/util/all-isl.h"
namespace RAJA {


template <typename T>
isl_union_map * lexico_ordering(isl_ctx * ctx, T numDims) {

  isl_space * dimSpace = isl_space_set_alloc(ctx, 0, numDims);


  isl_map * lexMap = isl_map_lex_lt(dimSpace);
  return isl_union_map_from_map(lexMap);

} //lexico_ordering

template <camp::idx_t LoopNum1, camp::idx_t LoopNum2>
int can_fuse(auto knl1, auto knl2) {

  isl_ctx * ctx = isl_ctx_alloc();


  isl_union_map * depRelation = data_dep_relation_from_knls<LoopNum1,LoopNum2>(ctx, knl1, knl2);

  isl_union_map * fsched1 = fused_schedule<LoopNum1>(ctx, knl1);
  isl_union_map * fsched2 = fused_schedule<LoopNum2>(ctx, knl2);

  isl_union_map * combinedFusedSchedule = isl_union_map_union(fsched1, fsched2);


  isl_union_map * applyRange = isl_union_map_apply_range(isl_union_map_copy(depRelation), isl_union_map_copy(combinedFusedSchedule));


  isl_union_map *applyDomain = isl_union_map_apply_domain(applyRange, isl_union_map_copy(combinedFusedSchedule));
  
  int numDimsInSchedule = 1 + std::max(knl1.numArgs, knl2.numArgs) * 2;
  isl_union_map * lexicoOrdering = lexico_ordering(ctx, numDimsInSchedule);

  isl_bool isLexicoPositive = isl_union_map_is_subset(applyDomain, lexicoOrdering);

 
  return isLexicoPositive;

} //can_fuse


template <typename T>
T intersect_union_sets(T set1) {
  return set1;
}

template <typename...Rest>
isl_union_set * intersect_union_sets(isl_union_set * set1, isl_union_set * set2, Rest...rest) {

  isl_union_set * tmpIntersection = isl_union_set_intersect(set1, set2);

  return intersect_union_sets(tmpIntersection,rest...);
} //intersect_union_sets


isl_union_map * pre_fused_schedule(isl_ctx * ctx, isl_union_set * fusedArea, auto knl) {

  isl_union_set * iterspace = iterspace_from_knl<0>(ctx, knl);
  
  isl_union_set * difference = isl_union_set_subtract(iterspace, fusedArea);

  isl_printer * p = isl_printer_to_file(ctx, stdout);

  std::cout << "\nDifference between fused area and iterspace\n";
  p = isl_printer_print_union_set(p, difference);
  std::cout << "\n";

  return NULL;

}
template <typename T>
T union_union_sets(T set1) {
  return set1;
}



template <typename...Rest>
isl_union_set * union_union_sets(isl_union_set * set1, isl_union_set * set2, Rest...rest) {

  isl_union_set * tmpUnion = isl_union_set_union(set1, set2);

  return union_union_sets(tmpUnion,rest...);
}

template <typename T>
T union_union_maps(T map1) {
  return map1;
}

template <typename...Rest>
isl_union_map * union_union_maps(isl_union_map * map1, isl_union_map * map2, Rest...rest) {
  isl_union_map * tempUnion = isl_union_map_union(map1, map2);

  return union_union_maps(tempUnion, rest...);
}
template <camp::idx_t... Is>
isl_union_map * union_union_map_tuple(auto mapTuple, camp::idx_seq<Is...>);

isl_union_map * union_union_map_tuple(auto mapTuple) {
  return union_union_map_tuple(mapTuple, idx_seq_for(mapTuple));
}
template <camp::idx_t... Is>
isl_union_map * union_union_map_tuple(auto mapTuple, camp::idx_seq<Is...>) {
  return union_union_maps(camp::get<Is>(mapTuple)...);
}


auto print_segment(auto segment) {
  auto l = *segment.begin();
  auto h = *segment.end();
  std::cout << "(" << l << "," << h << ")";
}
template <camp::idx_t CurrIdx, camp::idx_t MaxIdx>
auto print_segment_tuple(auto segmentTuple) {
  if constexpr (CurrIdx >= MaxIdx) {
    return;
  } else {
  print_segment(camp::get<CurrIdx>(segmentTuple));
  std::cout << " x ";
  print_segment_tuple<CurrIdx+1, MaxIdx>(segmentTuple);
  }
}



template <camp::idx_t...Dims>
auto pre_fused_range_segments_knl(auto fusedSegmentTuple, auto knl, camp::idx_seq<Dims...> dimSeq) {
  
  auto segmentTupleTuple = make_tuple(pre_fuse_kernels_segments<Dims>(knl.segments, fusedSegmentTuple, dimSeq)...);

  return segmentTupleTuple;

}

template <camp::idx_t...Is>
auto all_pre_fused_range_segments(auto fusedSegmentTuple, auto knlTuple, camp::idx_seq<Is...> knlSeq) {
  
  auto knl1 = camp::get<0>(knlTuple);
  auto knlDimSequence = camp::make_idx_seq_t<knl1.numArgs>{};

  auto rangeSegmentsKnl1 = pre_fused_range_segments_knl(fusedSegmentTuple, knl1, knlDimSequence);
  //auto rangeSegmentsKnl1 = make_tuple(pre_fuse_kernels_segments

  auto knl1segments1 = camp::get<0>(rangeSegmentsKnl1);

  auto knl1segments2 = camp::get<1>(rangeSegmentsKnl1);

  std::cout << "\nknl1 segments1:";
  print_segment_tuple<0,knl1.numArgs>(knl1segments1);
  std::cout << "\nknl1 segments2:";
  print_segment_tuple<0,knl1.numArgs>(knl1segments2);

  auto rangeSegmentsByKnl = make_tuple(pre_fused_range_segments_knl(fusedSegmentTuple, camp::get<Is>(knlTuple), knlDimSequence)...);

  std::cout << "\nknl1 segments1:";
  print_segment_tuple<0,knl1.numArgs>(camp::get<0>(camp::get<0>(rangeSegmentsByKnl)));
 std::cout << "\nknl1 segments2:";
  print_segment_tuple<0,knl1.numArgs>(camp::get<1>(camp::get<0>(rangeSegmentsByKnl)));
std::cout << "\nknl2 segments1:";
  print_segment_tuple<0,knl1.numArgs>(camp::get<0>(camp::get<1>(rangeSegmentsByKnl)));
 std::cout << "\nknl2 segments2:";
  print_segment_tuple<0,knl1.numArgs>(camp::get<1>(camp::get<1>(rangeSegmentsByKnl)));
  
  return rangeSegmentsByKnl;

}

template <camp::idx_t...Is>
isl_union_map * fusion_schedule(isl_ctx * ctx, auto knlTuple, camp::idx_seq<Is...> knlSeq) {


  isl_union_set * fusedArea = intersect_union_sets((iterspace_from_knl<0>(ctx, camp::get<Is>(knlTuple)))...);
 
  isl_printer * p = isl_printer_to_file(ctx, stdout);

  std::cout << "\nBounds for fused area of kernels\n";
  p = isl_printer_print_union_set(p, fusedArea); 
 
  auto  fusedRangeSegment = iterspace_to_segments<camp::get<0>(knlTuple).numArgs>(ctx, fusedArea);
  auto preFusedRangeSegments = all_pre_fused_range_segments(fusedRangeSegment, knlTuple, knlSeq);
   return NULL; 
} //fusion_schedule

//////////////

/*
template<camp::idx_t...Is>
isl_union_set * fused_iterspace(isl_ctx * ctx, auto knlTuple, camp::idx_seq<Is...>) {

  auto intersected = intersect_union_sets((iterspace_from_knl<0>(ctx, camp::get<Is>(knlTuple)))...);
 // fused_iterspace_helper(ctx, knlTuple);
  auto p = isl_printer_to_file(ctx, stderr);
  auto iterspaceTuple = make_tuple((iterspace_from_knl<Is+1>(ctx, camp::get<Is>(knlTuple)))...);

  std::cerr << "intersected iterspaces\n";
  p = isl_printer_print_union_set(p,intersected);
  std::cerr << "\n";

  auto unioned = union_union_sets((iterspace_from_knl<Is+1>(ctx, camp::get<Is>(knlTuple)))...);

  std::cerr << "unioned iterspaces\n";
  p = isl_printer_print_union_set(p, unioned);
  std::cout << "\n";

  auto both = isl_union_set_intersect(isl_union_set_copy(intersected), unioned);
p = isl_printer_print_union_set(p, both);

  return intersected;
} //fused_iterspace

*/


template <camp::idx_t DimNum>
auto iterspace_min(isl_union_set * iterspace) {
  isl_point * minPoint = isl_union_set_sample_point(isl_union_set_lexmin(isl_union_set_copy(iterspace)));
  return isl_val_get_num_si(isl_point_get_coordinate_val(minPoint, isl_dim_set, DimNum));
}

template <camp::idx_t DimNum>
auto iterspace_max(isl_union_set * iterspace) {
  isl_point * maxPoint = isl_union_set_sample_point(isl_union_set_lexmax(isl_union_set_copy(iterspace)));
  return isl_val_get_num_si(isl_point_get_coordinate_val(maxPoint, isl_dim_set, DimNum));
}

template <camp::idx_t KnlNum, camp::idx_t DimNum, camp::idx_t NumDims>
auto pre_constraint_set(isl_ctx * ctx, isl_union_set * fusedIterationSpace) {
  std::stringstream s;
  s << "{" << range_vector(NumDims, KnlNum) << " : " << "i" << DimNum << " < " << iterspace_min<DimNum>(fusedIterationSpace);


  s << "}";
  isl_union_set * constraintSet = isl_union_set_read_from_str(ctx, s.str().c_str());

  return constraintSet;
  

}

template <camp::idx_t KnlNum, camp::idx_t CurrDimIdx, camp::idx_t NumDims>
auto pre_iterspaces_from_knl_helper(isl_ctx * ctx, isl_union_set * fusedIterationSpace, isl_union_set * workingIterspace) {

  if constexpr (CurrDimIdx >= NumDims) {
    return make_tuple();
  } else {
    isl_union_set * constraintSet = pre_constraint_set<KnlNum, CurrDimIdx, NumDims>(ctx, isl_union_set_copy(fusedIterationSpace));
    isl_union_set * currDimIterspace = isl_union_set_intersect(isl_union_set_copy(workingIterspace), isl_union_set_copy(constraintSet));
    
    isl_union_set * newWorkingIterspace = isl_union_set_subtract(workingIterspace,constraintSet );
    auto restTuple = pre_iterspaces_from_knl_helper<KnlNum, CurrDimIdx + 1, NumDims>(ctx, fusedIterationSpace, newWorkingIterspace);

    return tuple_cat(make_tuple(currDimIterspace), restTuple);
   
  }

} //pre_iterspaces_from_knl_helper

template <camp::idx_t KnlNum>
auto pre_iterspaces_from_knl(isl_ctx * ctx, isl_union_set * fusedIterspace, auto knl) {
  
  auto startingIterspace = iterspace_from_knl<KnlNum>(ctx, knl);
  
  return pre_iterspaces_from_knl_helper<KnlNum,0,knl.numArgs>(ctx, fusedIterspace, startingIterspace);

}


template <camp::idx_t...Is>
auto pre_iterspaces_from_knls_helper(isl_ctx * ctx, isl_union_set * fusedIterspace, auto knlTuple, camp::idx_seq<Is...>) {
 return tuple_cat((pre_iterspaces_from_knl<Is+1>(ctx, fusedIterspace, camp::get<Is>(knlTuple)))...); 

}



auto pre_iterspaces_from_knls(isl_ctx * ctx, isl_union_set * fusedIterspace, auto knlTuple) {  
  return pre_iterspaces_from_knls_helper(ctx, fusedIterspace, knlTuple, idx_seq_for(knlTuple));
} //pre_iterspaces_from_knls

/*
 * post fused functions
 */
template <camp::idx_t KnlNum, camp::idx_t DimNum, camp::idx_t NumDims>
auto post_constraint_set(isl_ctx * ctx, isl_union_set * fusedIterationSpace) {
  std::stringstream s;
  s << "{" << range_vector(NumDims, KnlNum) << " : " << "i" << DimNum << " > " << iterspace_max<DimNum>(fusedIterationSpace);


  s << "}";
  std::cerr << "string stream for post_constaint_set: " << s.str() << "\n";
  isl_union_set * constraintSet = isl_union_set_read_from_str(ctx, s.str().c_str());

  return constraintSet;
  

}

template <camp::idx_t KnlNum, camp::idx_t CurrDimIdx, camp::idx_t NumDims>
auto post_iterspaces_from_knl_helper(isl_ctx * ctx, isl_union_set * fusedIterationSpace, isl_union_set * workingIterspace) {

  if constexpr (CurrDimIdx >= NumDims) {
    return make_tuple();
  } else {
    isl_union_set * constraintSet = post_constraint_set<KnlNum, CurrDimIdx, NumDims>(ctx, isl_union_set_copy(fusedIterationSpace));
    isl_union_set * currDimIterspace = isl_union_set_intersect(isl_union_set_copy(workingIterspace), isl_union_set_copy(constraintSet));
    
    isl_union_set * newWorkingIterspace = isl_union_set_subtract(workingIterspace,constraintSet );
    std::cerr << "post_iterspaces_from_knl_helper working iteration space:";
    isl_printer * p = isl_printer_to_file(ctx, stderr);
    isl_printer_print_union_set(p, newWorkingIterspace);
    std::cerr << "\n"; 
    auto restTuple = post_iterspaces_from_knl_helper<KnlNum, CurrDimIdx + 1, NumDims>(ctx, fusedIterationSpace, newWorkingIterspace);

    return tuple_cat(make_tuple(currDimIterspace), restTuple);
   
  }

} //post_iterspaces_from_knl_helper



template <camp::idx_t KnlNum, camp::idx_t CurrDimIdx, camp::idx_t NumDims>
auto post_iterspaces_starting_set_helper(isl_ctx * ctx, isl_union_set * fusedIterationSpace, isl_union_set * workingIterspace) {

  if constexpr (CurrDimIdx >= NumDims) {
    return workingIterspace;
  } else {
    isl_union_set * constraintSet = pre_constraint_set<KnlNum, CurrDimIdx, NumDims>(ctx, isl_union_set_copy(fusedIterationSpace));
    isl_union_set * currDimIterspace = isl_union_set_intersect(isl_union_set_copy(workingIterspace), isl_union_set_copy(constraintSet));
    
    isl_union_set * newWorkingIterspace = isl_union_set_subtract(workingIterspace,constraintSet );
    std::cerr << "post_iterspaces_starting_set_helper working iteration space:";
    isl_printer * p = isl_printer_to_file(ctx, stderr);
    isl_printer_print_union_set(p, newWorkingIterspace);
    std::cerr << "\n"; 
    return post_iterspaces_starting_set_helper<KnlNum, CurrDimIdx + 1, NumDims>(ctx, fusedIterationSpace, newWorkingIterspace); 
  }

} //post_iterspaces_starting_set_helper

template <camp::idx_t KnlNum>
auto post_iterspaces_starting_set(isl_ctx * ctx, isl_union_set * fusedIterspace, auto knl) {

  auto fullIterspace = iterspace_from_knl<KnlNum>(ctx, knl);

  return post_iterspaces_starting_set_helper<KnlNum,0,knl.numArgs>(ctx, fusedIterspace, fullIterspace);
  // all points greater than the minimum point in the fused part, where greater is 

  

} //post_iterspaces_starting_set


template <camp::idx_t KnlNum>
auto post_iterspaces_from_knl(isl_ctx * ctx, isl_union_set * fusedIterspace, auto knl) {
  
 
  auto startingIterspace = post_iterspaces_starting_set<KnlNum>(ctx, fusedIterspace, knl);
 
  return post_iterspaces_from_knl_helper<KnlNum,0,knl.numArgs>(ctx, fusedIterspace, startingIterspace);

}


template <camp::idx_t...Is>
auto post_iterspaces_from_knls_helper(isl_ctx * ctx, isl_union_set * fusedIterspace, auto knlTuple, camp::idx_seq<Is...>) {
 return tuple_cat((post_iterspaces_from_knl<Is+1>(ctx, fusedIterspace, camp::get<Is>(knlTuple)))...); 

} //post_iterspaces_from_knls_helper
auto post_iterspaces_from_knls(isl_ctx * ctx, isl_union_set * fusedIterspace, auto knlTuple) {

  
  return post_iterspaces_from_knls_helper(ctx, fusedIterspace, knlTuple, idx_seq_for(knlTuple));

} //post_iterspaces_from_knls


template <camp::idx_t KnlNum, camp::idx_t NumDims, camp::idx_t PartitionIndex>
isl_union_map * schedule_from_partition(isl_ctx * ctx, isl_union_set * iterspace) {
  
  std::cout << "Schedule from partition" << KnlNum << " " << NumDims << " "<< PartitionIndex << "\n"; 
  std::stringstream s;
  s << "{";
  s << range_vector(NumDims, KnlNum);
  s << " -> [";
  s << PartitionIndex;
  for(int i = 0; i < NumDims; i++) {
    s << ",";
    s << "i" << i << ",0";
    
  }
  s << "] }";

  isl_union_map * schedule = isl_union_map_read_from_str(ctx, s.str().c_str());
  
  return isl_union_map_intersect_domain(schedule,isl_union_set_copy(iterspace));


} //schedule_from_partition

template <camp::idx_t NumDims, camp::idx_t...Is>
auto pre_fused_schedules_from_partitions(isl_ctx * ctx, auto preFusedIterspaceTuple, camp::idx_seq<Is...>) {
  return make_tuple((schedule_from_partition<(Is/2)+1,NumDims,Is>(ctx, camp::get<Is>(preFusedIterspaceTuple)))...);

}

template <camp::idx_t NumDims, camp::idx_t...Is>
auto post_fused_schedules_from_partitions(isl_ctx * ctx, auto preFusedIterspaceTuple, camp::idx_seq<Is...>) {
  return make_tuple((schedule_from_partition<(Is/2)+1,NumDims,Is+(sizeof...(Is))+1>(ctx, camp::get<Is>(preFusedIterspaceTuple)))...);

}

template <camp::idx_t LoopNum, camp::idx_t NumDims, camp::idx_t StatementNum>
std::string fused_schedule_codomain_string() {

  std::stringstream s;
  s << "[";
  s << LoopNum;

  for(int i = 0; i < NumDims; i++) {
    s << ",i" << i << ",0";
  }

  s << "," << StatementNum;
  

  s << "]";

  return s.str();

}

//returns the internals of the string that describes the schedule for the fused bit. Does not include the {}
template <camp::idx_t CurrLoop, camp::idx_t NumLoops, camp::idx_t NumDims>
auto fused_schedule_from_partition_helper(isl_ctx * ctx, auto fusedIterspace) { 

  if constexpr (CurrLoop == NumLoops) {
    return make_tuple();
  } else {
    std::string thisLoopString = "{" + range_vector(NumDims, CurrLoop + 1) + " -> " + fused_schedule_codomain_string<(NumLoops*NumDims), NumDims, CurrLoop+1>() + ";}";
    isl_union_map * thisLoop = isl_union_map_read_from_str(ctx, thisLoopString.c_str());
    
    isl_union_map * thisLoopBounded = isl_union_map_intersect_domain(thisLoop,fusedIterspace);

    
    return tuple_cat(make_tuple(thisLoopBounded), fused_schedule_from_partition_helper<CurrLoop+1,NumLoops,NumDims>(ctx, fusedIterspace));
  }
  
}
template <camp::idx_t NumLoops, camp::idx_t NumDims>
auto fused_schedule_from_partition(isl_ctx * ctx, auto fusedIterspace) {
 
  std::stringstream s;
  
  auto partialSchedules = fused_schedule_from_partition_helper<0,NumLoops,NumDims>(ctx, fusedIterspace);  
  
  return union_union_map_tuple(partialSchedules, idx_seq_for(partialSchedules));
} //fused_schedule_from_partition

template <camp::idx_t NumDims, camp::idx_t...Is>
auto schedules_from_partitions_helper(isl_ctx * ctx, auto iterspaceTuple, camp::idx_seq<Is...>) {
 
  constexpr auto numPartitions = sizeof...(Is);
  
  auto preFusedIterspaces = slice_tuple<0,(numPartitions-1)/2>(iterspaceTuple);

  auto fusedIterspace = camp::get<(numPartitions-1)/2>(iterspaceTuple);

  auto postFusedIterspaces = slice_tuple<(numPartitions-1)/2+1, numPartitions>(iterspaceTuple);

  constexpr auto NumLoops = ((numPartitions - 1) / 2 ) / NumDims;
  
  isl_printer * p = isl_printer_to_file(ctx, stdout);
  std::cout << "Fused Iterspace in schedlue creation function\n";
  p = isl_printer_print_union_set(p,fusedIterspace);
  std::cout << "\n";

  auto preFusedSchedules = pre_fused_schedules_from_partitions<NumDims>(ctx, preFusedIterspaces, idx_seq_for(preFusedIterspaces));
  auto postFuseSchedules = post_fused_schedules_from_partitions<NumDims>(ctx, postFusedIterspaces, idx_seq_for(postFusedIterspaces));
  auto fusedSchedule = fused_schedule_from_partition<NumLoops, NumDims>(ctx, fusedIterspace);
 
  return tuple_cat(preFusedSchedules, make_tuple(fusedSchedule), postFuseSchedules);
  //return make_tuple((schedule_from_partition<Is/2,2,Is>(ctx, camp::get<Is>(iterspaceTuple)))...);
} //schedules_from_partitions_helper

template <typename...TupleType>
auto schedules_from_partitions(isl_ctx * ctx, camp::tuple<TupleType...> iterspaceTuple) {

  constexpr auto numPartitions = sizeof...(TupleType);

  constexpr auto numLoops = (numPartitions - 1) / 2;

  auto seq = idx_seq_for(iterspaceTuple);
  return schedules_from_partitions_helper<2>(ctx, iterspaceTuple, seq);

} //schedules_from_partitions


template <typename T>
auto max(T t1, T t2) {

  return std::max(t1,t2);
}// max
template <typename T, typename...Ts>
auto max(T t1, Ts...ts) {

  return std::max(t1, max(ts...));

}

template <typename T>
auto min(T t1, T t2) {

  return std::min(t1,t2);
}// min
template <typename T, typename...Ts>
auto min(T t1, Ts...ts) {

  return std::min(t1, min(ts...));

}

template <camp::idx_t DimNum, camp::idx_t...Is>
auto fused_iterspace_low_bound(isl_ctx *, auto iterspaceTuple, camp::idx_seq<Is...>) {
  return max((iterspace_min<DimNum>(camp::get<Is>(iterspaceTuple)))...);
}


template <camp::idx_t... Dims>
auto fused_iterspace_low_bounds(isl_ctx * ctx, auto iterspaceTuple, camp::idx_seq<Dims...>) {
  return make_tuple(fused_iterspace_low_bound<Dims>(ctx, iterspaceTuple, idx_seq_for(iterspaceTuple))...);
}

template <camp::idx_t DimNum, camp::idx_t...Is>
auto fused_iterspace_upper_bound(isl_ctx *, auto iterspaceTuple, camp::idx_seq<Is...>) {
  return min((iterspace_max<DimNum>(camp::get<Is>(iterspaceTuple)))...);
}


template <camp::idx_t... Dims>
auto fused_iterspace_upper_bounds(isl_ctx * ctx, auto iterspaceTuple, camp::idx_seq<Dims...>) {
  return make_tuple(fused_iterspace_upper_bound<Dims>(ctx, iterspaceTuple, idx_seq_for(iterspaceTuple))...);
}



template <camp::idx_t LoopNum, camp::idx_t CurrDim, camp::idx_t NumDims>
isl_union_set * bound_iterspace(isl_ctx * ctx, isl_union_set * set, auto lowBounds, auto highBounds) {

  if constexpr (CurrDim == NumDims) {
    return set;
  } else {
    auto lowBound = camp::get<CurrDim>(lowBounds);
    auto highBound = camp::get<CurrDim>(highBounds);

    std::stringstream s;
    s << "{" << range_vector(NumDims,LoopNum) << ": " << lowBound << "<= i" << CurrDim << " <= " << highBound << " }";
    std::cout << "bound_iterspace s: " << s.str() << "\n";
    isl_union_set * bounded = isl_union_set_read_from_str(ctx, s.str().c_str());

    isl_union_set * next = isl_union_set_intersect(set, bounded);
    isl_printer * p = isl_printer_to_file(ctx, stdout);

    std::cout << "After bounding\n";
    p = isl_printer_print_union_set(p,next);
   std::cout << "\n";
    return bound_iterspace<LoopNum, CurrDim+1, NumDims>(ctx, next, lowBounds, highBounds);

  }


}
template <typename...BoundTupleType, camp::idx_t...Is>
auto bound_iterspaces(isl_ctx * ctx, auto iterspaceTuple, camp::tuple<BoundTupleType...> lowBoundTuple, auto highBoundTuple, camp::idx_seq<Is...>) {

  
  return make_tuple((bound_iterspace<Is+1, 0,sizeof...(BoundTupleType)>(ctx, camp::get<Is>(iterspaceTuple), lowBoundTuple, highBoundTuple))...);
}


template <camp::idx_t...Is>
isl_union_set * union_union_sets_helper(auto unionSetTuple, camp::idx_seq<Is...>) {
  return union_union_sets(camp::get<Is>(unionSetTuple)...);
}
//returns the iteration space of the fused loop, which is across multipl loop domains
isl_union_set * fused_iterspace(isl_ctx * ctx, auto iterspaceTuple) {
  
  constexpr auto NumDims = camp::tuple_size<decltype(iterspaceTuple)>::value;

  auto lowBounds = fused_iterspace_low_bounds(ctx, iterspaceTuple, camp::make_idx_seq_t<NumDims>{});

  auto upperBounds = fused_iterspace_upper_bounds(ctx, iterspaceTuple, camp::make_idx_seq_t<NumDims>{});

  auto boundedIterspaces = bound_iterspaces(ctx, iterspaceTuple, lowBounds, upperBounds, idx_seq_for(iterspaceTuple));

  
  return union_union_sets_helper(boundedIterspaces, idx_seq_for(boundedIterspaces));
  return NULL;

}



} //namespace RAJA


#endif
