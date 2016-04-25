#ifndef __DOMAIN_TVIEW_H__
#define __DOMAIN_TVIEW_H__

#include<string>
#include<Kripke.h>


template<typename IdxLin, typename Perm, typename ... Idxs>
struct DLayout : public RAJA::Layout<IdxLin, Perm, Idxs...>{


  
  inline DLayout(Grid_Data &domain, int sdom_id) :
    RAJA::Layout<IdxLin, Perm, Idxs...>::Layout(
          domain.indexSize<Idxs>(sdom_id)...)
  {}

  template<typename ... ARGS>
  RAJA_HOST_DEVICE
  inline DLayout(ARGS ... args) :
    RAJA::Layout<IdxLin, Perm, Idxs...>(args...)
  {}

};


template<typename DataType, typename L>
struct DView {};

template<typename DataType, typename IdxLin, typename Perm, typename ... Idxs>
struct DView<DataType, DLayout<IdxLin, Perm, Idxs...>> : public RAJA::View<DataType, DLayout<IdxLin, Perm, Idxs...>> {

  inline DView(Grid_Data &domain, int sdom_id, DataType *ptr) :
    RAJA::View<DataType, DLayout<int, Perm, Idxs...>>(
        ptr,
        domain.indexSize<Idxs>(sdom_id)...)
  {}
};

#if 0

template<typename POL, typename IdxI, typename R, typename BODY>
RAJA_INLINE
void dForallN_expanded(Grid_Data &domain, int sdom_id, BODY const &body, R (BODY::*mf)(IdxI) const){

  RAJA::RangeSegment seg_i = domain.indexRange<IdxI>(sdom_id);

  // Call underlying forall, extracting ranges from domain
  RAJA::forallN<POL, IdxI>(seg_i, body);
}

template<typename POL, typename IdxI, typename IdxJ, typename R, typename BODY>
RAJA_INLINE
void dForallN_expanded(Grid_Data &domain, int sdom_id, BODY const &body, R (BODY::*mf)(IdxI, IdxJ) const){

  RAJA::RangeSegment seg_i = domain.indexRange<IdxI>(sdom_id);
  RAJA::RangeSegment seg_j = domain.indexRange<IdxJ>(sdom_id);

  // Call underlying forall, extracting ranges from domain
  RAJA::forallN<POL, IdxI, IdxJ>(seg_i, seg_j, body);
}



template<typename POL, typename IdxI, typename IdxJ, typename IdxK, typename R, typename BODY>
RAJA_INLINE
void dForallN_expanded(Grid_Data &domain, int sdom_id, BODY const &body, R (BODY::*mf)(IdxI, IdxJ, IdxK) const){

  RAJA::RangeSegment seg_i = domain.indexRange<IdxI>(sdom_id);
  RAJA::RangeSegment seg_j = domain.indexRange<IdxJ>(sdom_id);
  RAJA::RangeSegment seg_k = domain.indexRange<IdxK>(sdom_id);

  // Call underlying forall, extracting ranges from domain
  RAJA::forallN<POL, IdxI, IdxJ, IdxK>(seg_i, seg_j, seg_k, body);
}


template<typename POL, typename IdxI, typename IdxJ, typename IdxK, typename IdxL, typename R, typename BODY>
RAJA_INLINE
void dForallN_expanded(Grid_Data &domain, int sdom_id, BODY const &body, R (BODY::*mf)(IdxI, IdxJ, IdxK, IdxL) const){

  RAJA::RangeSegment seg_i = domain.indexRange<IdxI>(sdom_id);
  RAJA::RangeSegment seg_j = domain.indexRange<IdxJ>(sdom_id);
  RAJA::RangeSegment seg_k = domain.indexRange<IdxK>(sdom_id);
  RAJA::RangeSegment seg_l = domain.indexRange<IdxL>(sdom_id);

  // Call underlying forall, extracting ranges from domain
  RAJA::forallN<POL, IdxI, IdxJ, IdxK, IdxL>(seg_i, seg_j, seg_k, seg_l, body);
}


template<typename POLICY, typename BODY>
RAJA_INLINE 
void dForallN(Grid_Data &domain, int sdom_id, BODY body){
  dForallN_expanded<POLICY>(domain, sdom_id, body, &BODY::operator());
}

#else


#endif



template<typename POL, typename IdxI, typename BODY>
RAJA_INLINE
void dForallN(Grid_Data &domain, int sdom_id, BODY const &body){

  RAJA::RangeSegment seg_i = domain.indexRange<IdxI>(sdom_id);

  // Call underlying forall, extracting ranges from domain
  RAJA::forallN<POL, IdxI>(seg_i, body);
}

template<typename POL, typename IdxI, typename IdxJ, typename BODY>
RAJA_INLINE
void dForallN(Grid_Data &domain, int sdom_id, BODY const &body){

  RAJA::RangeSegment seg_i = domain.indexRange<IdxI>(sdom_id);
  RAJA::RangeSegment seg_j = domain.indexRange<IdxJ>(sdom_id);

  // Call underlying forall, extracting ranges from domain
  RAJA::forallN<POL, IdxI, IdxJ>(seg_i, seg_j, body);
}



template<typename POL, typename IdxI, typename IdxJ, typename IdxK, typename BODY>
RAJA_INLINE
void dForallN(Grid_Data &domain, int sdom_id, BODY const &body){

  RAJA::RangeSegment seg_i = domain.indexRange<IdxI>(sdom_id);
  RAJA::RangeSegment seg_j = domain.indexRange<IdxJ>(sdom_id);
  RAJA::RangeSegment seg_k = domain.indexRange<IdxK>(sdom_id);

  // Call underlying forall, extracting ranges from domain
  RAJA::forallN<POL, IdxI, IdxJ, IdxK>(seg_i, seg_j, seg_k, body);
}


template<typename POL, typename IdxI, typename IdxJ, typename IdxK, typename IdxL, typename BODY>
RAJA_INLINE
void dForallN(Grid_Data &domain, int sdom_id, BODY const &body){

  RAJA::RangeSegment seg_i = domain.indexRange<IdxI>(sdom_id);
  RAJA::RangeSegment seg_j = domain.indexRange<IdxJ>(sdom_id);
  RAJA::RangeSegment seg_k = domain.indexRange<IdxK>(sdom_id);
  RAJA::RangeSegment seg_l = domain.indexRange<IdxL>(sdom_id);

  // Call underlying forall, extracting ranges from domain
  RAJA::forallN<POL, IdxI, IdxJ, IdxK, IdxL>(seg_i, seg_j, seg_k, seg_l, body);
}



#endif



