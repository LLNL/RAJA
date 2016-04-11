/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see raja/README-license.txt
 */

#ifndef RAJA_TILE_HXX__
#define RAJA_TILE_HXX__

#include<RAJA/RAJA.hxx>

#define TILE_MIN(a, b) ((a) <= (b) ? (a) : (b))

namespace RAJA {

// Policy for no tiling
struct tile_none {};


// Policy to tile by given block size
template<int TileSize>
struct tile_fixed {};

template<typename TI, typename BODY>
void forall_tile(tile_none, TI const &is, BODY body){
  body(is);
}


template<int TileSize, typename BODY>
void forall_tile(tile_fixed<TileSize>, RAJA::RangeSegment const &is, BODY body){
  // tile loop
  Index_type i_begin = is.getBegin();
  Index_type i_end = is.getEnd();
  for(Index_type i0 = i_begin;i0 < i_end;i0 += TileSize){
  
    // Create a new tile
    Index_type i1 = TILE_MIN(i0+TileSize, i_end);
    RAJA::RangeSegment is_tile(i0, i1);
      
    // Pass tile index set to body        
    body(is_tile);
  }  
}




} // namespace RAJA

#endif

