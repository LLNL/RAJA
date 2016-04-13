/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */
  
#ifndef RAJA_forallN_tile_HXX__
#define RAJA_forallN_tile_HXX__

#include"config.hxx"
#include"int_datatypes.hxx"

namespace RAJA {


/******************************************************************
 *  ForallN tiling policies
 ******************************************************************/


// Policy for no tiling
struct tile_none {};


// Policy to tile by given block size
template<int TileSize>
struct tile_fixed {};


// Struct used to create a list of tiling policies
template<typename ... PLIST>
struct TileList{
  constexpr const static size_t num_loops = sizeof...(PLIST);
  typedef std::tuple<PLIST...> tuple;
};


// Tiling Policy
struct ForallN_Tile_Tag {};
template<typename TILE_LIST, typename NEXT=ForallN_Execute>
struct Tile {
  typedef ForallN_Tile_Tag PolicyTag;
  typedef NEXT NextPolicy;
  typedef TILE_LIST TilePolicy;
};



/******************************************************************
 *  Tiling mechanics
 ******************************************************************/


// Forward declaration so the apply_tile's can recurse into peel_tile
template<typename BODY, typename TilePolicy, int TIDX, typename PI, typename ... PREST>
RAJA_INLINE void forallN_peel_tile(BODY body, PI pi, PREST ... prest);



template<typename BODY, typename TilePolicy, int TIDX, typename PI, typename ... PREST>
RAJA_INLINE void forallN_apply_tile(tile_none, BODY body, PI pi, PREST ... prest){
  //printf("TIDX=%d: policy=tile_none\n", (int)TIDX);

  // Pass thru, so just bind the index set
  typedef ForallN_BindFirstArg<BODY, PI> BOUND;
  BOUND new_body(body, pi);

  // Recurse to the next policy
  forallN_peel_tile<BOUND, TilePolicy, TIDX+1>(new_body, prest...);
}


template<typename BODY, typename TilePolicy, int TIDX, int TileSize, typename PI, typename ... PREST>
RAJA_INLINE void forallN_apply_tile(tile_fixed<TileSize>, BODY body, PI pi, PREST ... prest){
  //printf("TIDX=%d: policy=tile_fixed<%d>\n", TIDX, TileSize);

  typedef ForallN_BindFirstArg<BODY, PI> BOUND;

  // tile loop
  Index_type i_begin = pi.getBegin();
  Index_type i_end = pi.getEnd();
  for(Index_type i0 = i_begin;i0 < i_end;i0 += TileSize){

    // Create a new tile
    Index_type i1 = std::min(i0+TileSize, i_end);
    PI pi_tile(RangeSegment(i0, i1));

    // Pass thru, so just bind the index set
    BOUND new_body(body, pi_tile);

    // Recurse to the next policy
    forallN_peel_tile<BOUND, TilePolicy, TIDX+1>(new_body, prest...);
  }
}


/******************************************************************
 *  forallN_policy(), tiling execution
 ******************************************************************/


template<typename NextPolicy, typename BODY, typename ... ARGS>
struct ForallN_NextPolicyWrapper {

  BODY body;

  explicit ForallN_NextPolicyWrapper(BODY b) : body(b) {}

  RAJA_INLINE
  void operator()(ARGS ... args) const {
    typedef typename NextPolicy::PolicyTag NextPolicyTag;

    forallN_policy<NextPolicy>(NextPolicyTag(), body, args...);
  }

};


template<typename BODY, typename TilePolicy, int TIDX>
RAJA_INLINE void forallN_peel_tile(BODY body){

  // Termination case just calls the tiling function that was constructed
  body();
}


template<typename BODY, typename TilePolicy, int TIDX, typename PI, typename ... PREST>
RAJA_INLINE void forallN_peel_tile(BODY body, PI pi, PREST ... prest){
  //printf("TIDX=%d\n", (int)TIDX);

  // Extract the tiling policy for loop nest TIDX
  using TP = typename std::tuple_element<TIDX, typename TilePolicy::tuple>::type;

  // Apply this index's policy, then recurse to remaining tile policies
  forallN_apply_tile<BODY, TilePolicy, TIDX>(TP(), body, pi, prest...);
}



/*!
 * \brief Tiling policy function.
 */
template<typename POLICY, typename BODY, typename ... PARGS>
RAJA_INLINE void forallN_policy(ForallN_Tile_Tag, BODY body, PARGS ... pargs){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  using TilePolicy = typename POLICY::TilePolicy;


  typedef ForallN_NextPolicyWrapper<NextPolicy, BODY, PARGS...> WRAPPER;
  WRAPPER wrapper(body);
  forallN_peel_tile<WRAPPER, TilePolicy, 0>(wrapper, pargs...);
}






} // namespace RAJA
  
#endif

