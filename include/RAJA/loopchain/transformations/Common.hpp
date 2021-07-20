// Contains common code for kernel transformations

#ifndef RAJA_LCCOMMON_HPP
#define RAJA_LCCOMMON_HPP

#include "RAJA/config.hpp"


namespace RAJA
{

// Function Declarations

template <typename...SegmentTupleTypes>
auto intersect_segment_tuples(camp::tuple<SegmentTupleTypes...> segmentTuples);

template <typename...SegmentTupleTypes>
auto intersect_segment_tuples(SegmentTupleTypes... segmentTuples);



// Function Definitions

// given a tuple of segments, returns one segment that contained by all the segments within the tuple
template <typename...SegmentTypes, camp::idx_t...Is>
auto intersect_segments(camp::tuple<SegmentTypes...> segments, camp::idx_seq<Is...>) {
  auto highestLow = vmax((*((camp::get<Is>(segments)).begin()))...);
  auto lowestHigh = vmin((*((camp::get<Is>(segments)).end()))...);

  return RangeSegment(highestLow, lowestHigh);
}

template <typename...SegmentTypes>
auto intersect_segments(camp::tuple<SegmentTypes...> segments) {
  return intersect_segments(segments, idx_seq_for(segments));
}

template <typename...SegmentTupleTypes, camp::idx_t...NumTuples, camp::idx_t...NumDims>
auto intersect_segment_tuples(camp::tuple<SegmentTupleTypes...> segmentTuples, 
                              camp::idx_seq<NumTuples...>,
                              camp::idx_seq<NumDims...>) {
  auto groupedByDim =  tuple_zip(segmentTuples);
  return make_tuple(intersect_segments(camp::get<NumDims>(groupedByDim))...);
}



template <typename...SegmentTupleTypes, camp::idx_t...Is>
auto intersect_segment_tuples(camp::tuple<SegmentTupleTypes...> segmentTuples, camp::idx_seq<Is...> seq){
  return intersect_segment_tuples(segmentTuples, seq, idx_seq_for(camp::get<0>(segmentTuples)));
}
// For an arbitrary number of segment tuples of the same dimensionality,
// returns a segment tuple for the intersection of the segment tuples.
// This is the shared iteration space of the kernels with the provided segment tuple
template <typename...SegmentTupleTypes>
auto intersect_segment_tuples(camp::tuple<SegmentTupleTypes...> segmentTuples) {
  return intersect_segment_tuples(segmentTuples, idx_seq_for(segmentTuples));
}


template <typename...SegmentTupleTypes>
auto intersect_segment_tuples(SegmentTupleTypes... segmentTuples) {
  return intersect_segment_tuples(make_tuple(segmentTuples...));
}

// Common ISL Code Declarations


template <typename KernelType>
isl_union_set * knl_iterspace(isl_ctx * ctx, KernelType knl, camp::idx_t id);

template <typename KernelType>
isl_union_map * knl_read_relation(isl_ctx * ctx, KernelType knl, camp::idx_t id);

template <typename KernelType>
isl_union_map * knl_write_relation(isl_ctx * cxt, KernelType knl, camp::idx_t id);

template <typename KernelType1, typename KernelType2>
isl_union_map * flow_dep_relation(isl_ctx * ctx,
                                  KernelType1 knl1, KernelType2 knl2,
                                  camp::idx_t id1, camp::idx_t id2);

template <typename KernelType1, typename KernelType2>
isl_union_map * anti_dep_relation(isl_ctx * ctx,
                                  KernelType1 knl1, KernelType2 knl2,
                                  camp::idx_t id1, camp::idx_t id2);

template <typename KernelType1, typename KernelType2>
isl_union_map * output_dep_relation(isl_ctx * ctx,
                                  KernelType1 knl1, KernelType2 knl2,
                                  camp::idx_t id1, camp::idx_t id2);

template <typename KernelType1, typename KernelType2>
isl_union_map * data_dep_relation(isl_ctx * ctx,
                                  KernelType1 knl1, KernelType2 knl2,
                                  camp::idx_t id1, camp::idx_t id2);


// Common ISL Code Definitions

// gives each view a unique identifier for its use in access relations
template <typename AccessType>
auto get_array_name(AccessType access) {
  static std::map<void*, camp::idx_t> idMap = {};
  static camp::idx_t counter = 0;
  auto ptr = access.view;

  
  auto search = idMap.find(ptr);
  if(search == idMap.end()) {
    idMap[ptr] = counter;
    counter++;
  }

  return "arr" + std::to_string(idMap[ptr]);
}

template <typename SegmentType>
std::string segment_to_constraints(SegmentType segment, camp::idx_t idx) {
  auto low = *segment.begin();
  auto high = *segment.end();

  std::stringstream s;
  s << "i" << idx << " >= " << low << " and " << "i" << idx << " < " << high;
  std::string str = s.str();

  return str;
}


template <camp::idx_t I, camp::idx_t...Is, typename... SegmentTypes>
auto segment_tuple_to_constraints(camp::tuple<SegmentTypes...> segmentTuple, camp::idx_seq<I, Is...>) {
   std::string rest;
  if constexpr (sizeof...(Is) > 0) {
    rest = segment_tuple_to_constraints<Is...>(segmentTuple, camp::idx_seq<Is...>{});
    return segment_to_constraints(camp::get<I>(segmentTuple), I) + " and " + rest;
  } else {
    return segment_to_constraints(camp::get<I>(segmentTuple), I);
  }
}



//returns the string for the codomain vector with numElements dimension. For example,
// 2 return "[i0,i1]" and 4 returns "[i0,i1,i2,i3]"
template <typename IdxType>
std::string codomain_vector(IdxType numElements, camp::idx_t loopNum) {

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
}

// returns a isl union set describing the iteration space of a kernel. uses id to identify the iteration space
template <typename KernelType>
isl_union_set * knl_iterspace(isl_ctx * ctx, KernelType knl, camp::idx_t id) {
  auto segments = knl.segments;

  auto bounds = segment_tuple_to_constraints(segments, idx_seq_for(segments));

  auto codomainString = codomain_vector(knl.numArgs, id);
 
  auto iterspaceString = "{ " + codomainString + " : " + bounds + " }";

  isl_union_set * iterspace = isl_union_set_read_from_str(ctx, iterspaceString.c_str());

  return iterspace;
}


// returns a relation describing the mapping between the iteration space of a kernel and its read accesses
template <typename KernelType>
isl_union_map * read_relation(isl_ctx * ctx, KernelType knl, camp::idx_t id) {
  auto accesses = knl.execute_symbolically();
 
  std::stringstream readString;
  readString << "{";

  for(auto access : accesses) {
    if(access.isRead) {
      auto name = get_array_name(access);
      auto accessString = access.access_string();
      readString << "\t" << codomain_vector(knl.numArgs, id) << " -> " << name << "[" << accessString << "];";
    }
  }

  readString << "}";
  
  isl_union_map * readRelation = isl_union_map_read_from_str(ctx, readString.str().c_str());
  isl_union_map * readRelationForLoop = isl_union_map_intersect_domain(readRelation, knl_iterspace(ctx, knl, id));

  return readRelationForLoop;
}

// returns a relation describing the mapping between the iteration space and its write accesses
template <typename KernelType>
isl_union_map * write_relation(isl_ctx * ctx, KernelType knl, camp::idx_t id) {
  auto accesses = knl.execute_symbolically();

  std::stringstream writeString;
  writeString << "{";

  for(auto access : accesses) {
    if(access.isWrite) {
      auto name = get_array_name(access);
      auto accessString = access.access_string();
      writeString << "\t" << codomain_vector(knl.numArgs, id) << " -> " << name << "[" << accessString << "];";
    }
  }

  writeString << "}";

  isl_union_map * writeRelation = isl_union_map_read_from_str(ctx, writeString.str().c_str());
  isl_union_map * writeRelationForLoop = isl_union_map_intersect_domain(writeRelation, knl_iterspace(ctx, knl, id));

  return writeRelationForLoop;
}

// returns a relation between the iteration spaces of the two kernels describing the flow dependences (raw)
template <typename KernelType1, typename KernelType2>
isl_union_map * flow_dep_relation(isl_ctx * ctx,
                                  KernelType1 knl1, KernelType2 knl2,
                                  camp::idx_t id1, camp::idx_t id2) {
  auto writes = write_relation(ctx, knl1, id1);
  auto reads = read_relation(ctx, knl2, id2);

  auto readsInverse = isl_union_map_reverse(reads);
  isl_union_map * readsAfterWrites = isl_union_map_apply_range(writes, readsInverse);

  return readsAfterWrites;
}


// returns a relation between the tieration spaces of the kernels describing the anti dependences (war)
template <typename KernelType1, typename KernelType2>
isl_union_map * anti_dep_relation(isl_ctx * ctx,
                                  KernelType1 knl1, KernelType2 knl2,
                                  camp::idx_t id1, camp::idx_t id2) {

  auto reads = read_relation(ctx, knl1, id1);
  auto writes = write_relation(ctx, knl2, id2);

  auto writesInverse = isl_union_map_reverse(writes);
  isl_union_map * writesAfterReads = isl_union_map_apply_range(reads, writesInverse);

  return writesAfterReads;
}

// returns a relation between the iteration spaces of the kernels describing the output dependences (waw)
template <typename KernelType1, typename KernelType2>
isl_union_map * output_dep_relation(isl_ctx * ctx,
                                  KernelType1 knl1, KernelType2 knl2,
                                  camp::idx_t id1, camp::idx_t id2) {
  auto src = write_relation(ctx, knl1, id1);
  auto dst = write_relation(ctx, knl2, id2);

  auto dstInverse = isl_union_map_reverse(dst);
  isl_union_map * srcToDst = isl_union_map_apply_range(src, dstInverse);

  return srcToDst;
}

// returns a relation between the iteration spaces of the kernels describing all data dependences between the kernels
template <typename KernelType1, typename KernelType2>
isl_union_map * data_dep_relation(isl_ctx * ctx,
                                  KernelType1 knl1, KernelType2 knl2,
                                  camp::idx_t id1, camp::idx_t id2) {
  auto raw = flow_dep_relation(ctx, knl1, knl2, id1, id2);
  auto war = anti_dep_relation(ctx, knl1, knl2, id1, id2);
  auto waw = output_dep_relation(ctx, knl1, knl2, id1, id2);

  auto temp = isl_union_map_union(raw, war);
  auto dataDeps = isl_union_map_union(temp, waw);

  return dataDeps;
}




} //namespace RAJA
#endif

