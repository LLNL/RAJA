// Contains functions that extract isl polyhedral sets from kernel objects.

#ifndef RAJA_LC_knlconv_HPP
#define RAJA_LC_knlconv_HPP

#include "RAJA/util/all-isl.h"
#include <iostream>
#include <sstream>

#include "RAJA/loopchain/KernelWrapper.hpp"
#include "RAJA/loopchain/Utils.hpp"
namespace RAJA {

template <typename T>
T get_array_name(SymAccess a);

template <camp::idx_t Dim>
auto points_to_segment(isl_point * minPoint, isl_point * maxPoint) {

  auto _minVal = isl_point_get_coordinate_val(minPoint, isl_dim_set, Dim); 
  auto _maxVal = isl_point_get_coordinate_val(maxPoint, isl_dim_set, Dim); 
  
  auto minVal = isl_val_get_num_si(_minVal);
  auto maxVal = isl_val_get_num_si(_maxVal);
  
  std::cout << "Creating segment from " << minVal << "to " << maxVal << "\n";
  return RangeSegment(minVal, maxVal);
} //points_to_segment

template <camp::idx_t...Dims>
auto points_to_segments(isl_point * minPoint, isl_point * maxPoint, camp::idx_seq<Dims...>) {

  return make_tuple((points_to_segment<Dims>(minPoint, maxPoint))...);

}// points_to_segments

template <camp::idx_t NumDims>
auto iterspace_to_segments(isl_ctx * ctx, isl_union_set * iterspace) {
  
  
  isl_printer * p = isl_printer_to_file(ctx, stdout);
  
  std::cout << "\nFinding segments for iterspace:";
  p = isl_printer_print_union_set(p,iterspace);
  std::cout << "\n";
  isl_union_set * minSet = isl_union_set_lexmin(isl_union_set_copy(iterspace));

  
  std::cout << "\nLexicographical minimum of iterspace: ";
  p = isl_printer_print_union_set(p, minSet);
  std::cout << "\n";
  
  isl_union_set * maxSet = isl_union_set_lexmax(iterspace);

  std::cout << "\nLex max of iterspace: ";
  p = isl_printer_print_union_set(p,maxSet);
  std::cout << "\n";

  isl_point * minPoint = isl_union_set_sample_point(minSet);
  isl_point * maxPoint = isl_union_set_sample_point(maxSet);

  std::cout << "\nMin and Max /points/ of iterspace:\n";
  p = isl_printer_print_point(p, minPoint);
  std::cout << "\n";
  p =  isl_printer_print_point(p, maxPoint);
  std::cout << "\n";

  auto segments = points_to_segments(minPoint, maxPoint, camp::make_idx_seq_t<NumDims>{});
   
  return segments;

} //iterspace_to_segments

template <camp::idx_t Idx, typename Segment>
std::string bound_from_segment(Segment segment) {
  
  auto low = *segment.begin();
  auto high = *segment.end();

  std::stringstream s;
  s << "i" << Idx << " >= " << low << " and " << "i" << Idx << " < " << high;
  std::string str = s.str();

  return str;
}

template <typename T, camp::idx_t I>
std::string bounds_from_segment(T segmentTuple, camp::idx_seq<I>) {

  return bound_from_segment<I>(camp::get<I>(segmentTuple));
  
}

template <typename T, camp::idx_t I, camp::idx_t...Is>
std::string bounds_from_segment(T segmentTuple, camp::idx_seq<I, Is...>) {
  std::string rest;
  if constexpr (sizeof...(Is) > 0) {
    rest = bounds_from_segment<Is...>(segmentTuple, camp::idx_seq<Is...>{});
    return bound_from_segment<I>(camp::get<I>(segmentTuple)) + " and " + rest;
  } else {
    return bound_from_segment<I>(camp::get<I>(segmentTuple));
  }
}


//returns the string for the range vector with numElements dimension. For example,
// 2 return "[i0,i1]" and 4 returns "[i0,i1,i2,i3]"
template <typename T>
std::string range_vector(T numElements, camp::idx_t loopNum) {

  std::stringstream vec;
  vec << "L" << loopNum << "[";

  for(int i = 0; i < numElements; i++) { 
    vec << "i" << i;
    if(i < numElements - 1) {
      vec << ",";
    }
  }

  vec << "]";

  return vec.str();
} //range_vector

//Returns the iteration space polyhedron of knl. 
//Does so without the loop number in the chain, 
//so that must be prepended if necessary.
template <typename T, camp::idx_t LoopNum>
isl_union_set * iterspace_from_knl(isl_ctx * ctx, T knl) {


  auto segments = knl.segments;

  auto segmentSeq = idx_seq_for(segments);

  auto bounds = bounds_from_segment(segments, segmentSeq);

  auto rangeVector = range_vector(knl.numArgs, LoopNum);  

  auto iterspaceString = "{ " + rangeVector + " : " + bounds + " }";
 
  isl_union_set * iterspace = isl_union_set_read_from_str(ctx, iterspaceString.c_str());

  return iterspace; 

} // iterspace_from_knl

template <typename T, camp::idx_t LoopNum>
isl_union_map * read_relation_from_knl(isl_ctx * ctx, T knl) {

  //isl_printer* p = isl_printer_to_file(ctx, stdout);

  auto accesses = knl.execute_symbolically();
 
  std::stringstream readString;

  readString << "{";

  for(auto access : accesses) {
    if(access.isRead) {

      auto name = get_array_name(access);
      auto accessString = access.access_string();
      readString << "\t" << range_vector(knl.numArgs, LoopNum) << " -> " << name << "[" << accessString << "];";
    }
  }

  readString << "}";

  
  isl_union_map * readRelation = isl_union_map_read_from_str(ctx, readString.str().c_str());

  isl_union_map * readRelationForLoop = isl_union_map_intersect_domain(readRelation, iterspace_from_knl<LoopNum>(ctx, knl));

  return readRelationForLoop;

  
} // read_relation_from_knl

template <typename T, camp::idx_t LoopNum>
isl_union_map * write_relation_from_knl(isl_ctx * ctx, T knl) {

  //isl_printer* p = isl_printer_to_file(ctx, stdout);

  auto accesses = knl.execute_symbolically();
 
  std::stringstream writeString;

  writeString << "{";

  for(auto access : accesses) {
    if(access.isWrite) {

      auto name = get_array_name(access);
      auto accessString = access.access_string();
      writeString << "\t" << range_vector(knl.numArgs, LoopNum) << " -> " << name << "[" << accessString << "];";
    }
  }

  writeString << "}";

  
  isl_union_map * writeRelation = isl_union_map_read_from_str(ctx, writeString.str().c_str());

  isl_union_map * writeRelationForLoop = isl_union_map_intersect_domain(writeRelation, iterspace_from_knl<LoopNum>(ctx, knl));

  return writeRelationForLoop;

  
} // read_relation_from_knl


//returns a relation between the iteration spaces of the two knls mapping instances of knl1 to 
//instances of knl2 where knl1 instance writes and knl2 instance reads the same location
template <camp::idx_t LoopNum1, camp::idx_t LoopNum2, typename T1, typename T2>
isl_union_map * flow_dep_relation_from_knls(isl_ctx * ctx,
                                            T1 knl1, 
                                            T2 knl2) {

  isl_union_map * knl1Writes = write_relation_from_knl<LoopNum1>(ctx, knl1);
  isl_union_map * knl2Reads = read_relation_from_knl<LoopNum2>(ctx, knl2);

  //We want a map from I1 to I2 where i1,i2 iff knl1WriteRelation(i1) and knl2ReadRelation(i2) have shared elements
 
  isl_union_map * readsInverse = isl_union_map_reverse(knl2Reads);

  //We apply the range of knl1Writes to the inverse of knl2Reads. 
  //The range of knl1Writes is the data accessed, which is the domain of the knl2Reads inverse
  //This basically feeds the data locations into the inversed knl2Reads, 
  //which gives out the iterations of knl2 that made the accesses
  isl_union_map * readsAfterWrites = isl_union_map_apply_range(knl1Writes, readsInverse);
  return readsAfterWrites;
} // flow_dep_relation_from_knls


//returns a relation between the iteration spaces of the two knls mapping instances of knl1 to 
//instances of knl2 where knl1 instance reads and knl2 instance writes the same location
template <camp::idx_t LoopNum1, camp::idx_t LoopNum2, typename T1, typename T2 >
isl_union_map * anti_dep_relation_from_knls(isl_ctx * ctx,
                                            T1 knl1, 
                                            T2 knl2) {

  isl_union_map * knl1Reads = read_relation_from_knl<LoopNum1>(ctx, knl1);
  isl_union_map * knl2Writes = write_relation_from_knl<LoopNum2>(ctx, knl2);

 
  isl_union_map * writesInverse = isl_union_map_reverse(knl2Writes);
  isl_union_map * writesAfterReads = isl_union_map_apply_range(knl1Reads, writesInverse);
  return writesAfterReads;
} // anti_dep_relation_from_knls

//returns a relation between the iteration spaces of the two knls mapping instances of knl1 to 
//instances of knl2 where knl1 instance writes and knl2 instance writes the same location
template <camp::idx_t LoopNum1, camp::idx_t LoopNum2,  typename T1, typename T2 >
isl_union_map * output_dep_relation_from_knls(isl_ctx * ctx,
                                            T1 knl1, 
                                            T2 knl2) {

  isl_union_map * knl1Writes = write_relation_from_knl<LoopNum1>(ctx, knl1);
  isl_union_map * knl2Writes = write_relation_from_knl<LoopNum2>(ctx, knl2);

 
  isl_union_map * writesInverse = isl_union_map_reverse(knl2Writes);
  isl_union_map * writesAfterWrites = isl_union_map_apply_range(knl1Writes, writesInverse);
  return writesAfterWrites;
} // output_dep_relation_from_knls

//returns a relation between the iteration spaces of the two knls mapping instances of knl1 to instances of knl2 where there is a data dependence of some sort between the two instances.
template <camp::idx_t LoopNum1, camp::idx_t LoopNum2, typename T1, typename T2 >
isl_union_map * data_dep_relation_from_knls(isl_ctx * ctx,
                                            T1 knl1, 
                                            T2 knl2) {

  isl_union_map * raw = flow_dep_relation_from_knls<LoopNum1,LoopNum2>(ctx, knl1, knl2);
  isl_union_map * war = anti_dep_relation_from_knls<LoopNum1,LoopNum2>(ctx, knl1, knl2);
  isl_union_map * waw = output_dep_relation_from_knls<LoopNum1,LoopNum2>(ctx, knl1, knl2);

  isl_union_map * partialUnion = isl_union_map_union(war, waw); 
  isl_union_map * depUnion = isl_union_map_union(raw, partialUnion);

  return depUnion;

} // data_dep_relation_from_knls


//returns the range string for normal execution of a loop with numdims that is loop loopNum in teh chain
template <typename T>
std::string original_schedule_range(T numDims, T loopNum) {

  std::stringstream s;

  s << "[";
  s << loopNum << ",";

  for(int i = 0; i < numDims; i++) {
    s << "i" << i << ",0";
    if( i != numDims - 1) {
      s << ",";
    }
  }
  
  s << "]";

  return s.str();
} // original_schedule_domain

// Returns the schedule for the loop executed sequentially. For example,
// a double nested loop will give [i,j] -> [loopNum, i, 0, j, 0]
template <typename T, camp::idx_t LoopNum>
isl_union_map * original_schedule(isl_ctx * ctx, T knl) {

  isl_union_set * iterspace = iterspace_from_knl<LoopNum>(ctx, knl);

  std::string scheduleRange = original_schedule_range(knl.numArgs, LoopNum);

  
  
  std::string scheduleRelation = "{" + range_vector(knl.numArgs, LoopNum) + " -> " + scheduleRange + "}";

  isl_union_map * scheduleNoBounds = isl_union_map_read_from_str(ctx, scheduleRelation.c_str());

  isl_union_map * scheduleWithBounds = isl_union_map_intersect_domain(scheduleNoBounds, iterspace);

  return scheduleWithBounds;
} //original_schedule


//returns the range string for normal execution of a loop with numdims that is loop loopNum in teh chain
template <typename T>
std::string fused_schedule_range(T numDims, T loopNum) {

  std::stringstream s;

  s << "[";
 
  for(int i = 0; i < numDims; i++) {
    s << "0, i" << i << ",";
  }
  
  s << loopNum << "]";

  return s.str();
} // fused_schedule_domain


// Returns the schedule for the loop executed sequentially. For example,
// a double nested loop will give [i,j] -> [0, i, 0, j, loopNum]
template <typename T, camp::idx_t LoopNum>
isl_union_map * fused_schedule(isl_ctx * ctx, T knl) {

  isl_union_set * iterspace = iterspace_from_knl<LoopNum>(ctx, knl);

  std::string scheduleRange = fused_schedule_range(knl.numArgs, LoopNum);

  
  
  std::string scheduleRelation = "{" + range_vector(knl.numArgs, LoopNum) + " -> " + scheduleRange + "}";

  isl_union_map * scheduleNoBounds = isl_union_map_read_from_str(ctx, scheduleRelation.c_str());

  isl_union_map * scheduleWithBounds = isl_union_map_intersect_domain(scheduleNoBounds, iterspace);
   
  
  return scheduleWithBounds;
} //fused_schedule


template <camp::idx_t CurrKnl1, camp::idx_t CurrKnl2, camp::idx_t NumKnls, typename...KnlTypes>
isl_union_map * deprel_from_knls_helper(isl_ctx * ctx, camp::tuple<KnlTypes...> knlTuple) {
  if constexpr (CurrKnl1 > NumKnls) {
    return isl_union_map_read_from_str(ctx, "{}"); 
  }

  else if constexpr (CurrKnl2 > NumKnls) {
    return deprel_from_knls_helper<CurrKnl1 + 1, CurrKnl1 + 1, NumKnls>(ctx, knlTuple);
  } else {
    isl_union_map * thisPair = data_dep_relation_from_knls<CurrKnl1, CurrKnl2>(
      ctx, camp::get<CurrKnl1-1>(knlTuple), camp::get<CurrKnl2-1>(knlTuple));
    
    

    return isl_union_map_union(thisPair, 
      deprel_from_knls_helper<CurrKnl1, CurrKnl2 + 1, NumKnls>(ctx, knlTuple));
  }
}

template <typename T, camp::idx_t...Is>
auto dependence_relation_from_kernels(isl_ctx * ctx, T knlTuple, camp::idx_seq<Is...> knl) {
  constexpr auto NumKnls = sizeof...(Is); 
  return deprel_from_knls_helper<1,1,NumKnls>(ctx, knlTuple);
} //dependence_relation_from_kernels

template <typename T>
auto dependence_relation_from_kernels(isl_ctx * ctx, T knlTuple) {
  return dependence_relation_from_kernels(ctx, knlTuple, idx_seq_for(knlTuple));
} //dependence_relation_from_kernels


template <typename T, camp::idx_t...Is>
auto original_schedule_from_kernels(isl_ctx * ctx, T knlTuple, camp::idx_seq<Is...> knl) {

  //First, collect all dependences among the kernels.  
  
  return NULL;
}

template <typename...KnlTypes>
auto original_schedule_from_kernels(isl_ctx * ctx, camp::tuple<KnlTypes...> knlTuple) {
  return original_schedule_from_kernels(ctx, knlTuple, idx_seq_for(knlTuple));
}

template <typename...KnlTypes, camp::idx_t...Is>
auto equal_iterspaces(camp::tuple<KnlTypes...> knlTuple, camp::idx_seq<Is...>) {
  isl_ctx * ctx = isl_ctx_alloc();
  auto iterspaces = make_tuple(iterspace_from_knl<0>(ctx,camp::get<Is>(knlTuple))...);
  
  auto areEqualToFirst = make_tuple(isl_union_set_is_equal(camp::get<0>(iterspaces), camp::get<Is>(iterspaces))...);

  auto areAllEqualToFirst = (camp::get<Is>(areEqualToFirst) && ...);

  return areAllEqualToFirst;
  
}
template <typename...KnlTypes>
auto equal_iterspaces(camp::tuple<KnlTypes...> knlTuple) {
  return equal_iterspaces(knlTuple, idx_seq_for(knlTuple));
}
} //namespace RAJA
#endif
