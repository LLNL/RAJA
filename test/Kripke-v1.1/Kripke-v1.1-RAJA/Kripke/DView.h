#ifndef __DOMAIN_TVIEW_H__
#define __DOMAIN_TVIEW_H__

#include<string>
#include<RAJA/RangeSegment.hxx>
#include<RAJA/IndexValue.hxx>
#include<RAJA/View.hxx>
#include<RAJA/Forall.hxx>


template<typename DataType, typename Layout>
struct DView1d : public RAJA::View1d<DataType, Layout> {

  typedef typename Layout::Permutation Permutation;
  typedef typename Layout::IndexI IndexI;

  inline DView1d(DataType *ptr, Grid_Data *domain, int sdom_id) :
    RAJA::View1d<DataType, Layout>::View1d(
        ptr,
        domain->indexSize<IndexI>(sdom_id))
  {}
};

template<typename DataType, typename Layout>
struct DView2d : public RAJA::View2d<DataType, Layout> {

  typedef typename Layout::Permutation Permutation;
  typedef typename Layout::IndexI IndexI;
  typedef typename Layout::IndexJ IndexJ;

  inline DView2d(DataType *ptr, Grid_Data *domain, int sdom_id) :
    RAJA::View2d<DataType, Layout>::View2d(
        ptr,
        domain->indexSize<IndexI>(sdom_id),
        domain->indexSize<IndexJ>(sdom_id))
  {}
};

template<typename DataType, typename Layout>
struct DView3d : public RAJA::View3d<DataType, Layout> {

  typedef typename Layout::Permutation Permutation;
  typedef typename Layout::IndexI IndexI;
  typedef typename Layout::IndexJ IndexJ;
  typedef typename Layout::IndexK IndexK;

  inline DView3d(DataType *ptr, Grid_Data *domain, int sdom_id) :
    RAJA::View3d<DataType, Layout>::View3d(
        ptr,
        domain->indexSize<IndexI>(sdom_id),
        domain->indexSize<IndexJ>(sdom_id),
        domain->indexSize<IndexK>(sdom_id))
  {}
};

template<typename DataType, typename Layout>
struct DView4d : public RAJA::View4d<DataType, Layout> {

  typedef typename Layout::Permutation Permutation;
  typedef typename Layout::IndexI IndexI;
  typedef typename Layout::IndexJ IndexJ;
  typedef typename Layout::IndexK IndexK;
  typedef typename Layout::IndexL IndexL;

  inline DView4d(DataType *ptr, Grid_Data *domain, int sdom_id) :
    RAJA::View4d<DataType, Layout>::View4d(
        ptr,
        domain->indexSize<IndexI>(sdom_id),
        domain->indexSize<IndexJ>(sdom_id),
        domain->indexSize<IndexK>(sdom_id),
        domain->indexSize<IndexL>(sdom_id))
  {}
};


/**
 * Wrapper around Layout1d that provides accessors to Index sizes
 */
template<typename Perm, typename IdxI, typename IdxLin = int>
struct DLayout1d : public RAJA::Layout1d<Perm, IdxI, IdxLin>{

  inline DLayout1d(Grid_Data *domain, int sdom_id) :
    RAJA::Layout1d<Perm, IdxI, IdxLin>::Layout1d(
          domain->indexSize<IdxI>(sdom_id))
  {}

  inline DLayout1d(int ni) :
    RAJA::Layout1d<Perm, IdxI, IdxLin>::Layout1d(ni)
  {}

};


/**
 * Wrapper around Layout2d that provides accessors to Index sizes
 */
template<typename Perm, typename IdxI, typename IdxJ, typename IdxLin = int>
struct DLayout2d : public RAJA::Layout2d<Perm, IdxI, IdxJ, IdxLin>{

  inline DLayout2d(Grid_Data *domain, int sdom_id) :
    RAJA::Layout2d<Perm, IdxI, IdxJ, IdxLin>::Layout2d(
          domain->indexSize<IdxI>(sdom_id),
          domain->indexSize<IdxJ>(sdom_id))
  {}

  inline DLayout2d(int ni, int nj) :
    RAJA::Layout2d<Perm, IdxI, IdxJ, IdxLin>::Layout2d(ni, nj)
  {}

};

/**
 * Wrapper around Layout3d that provides accessors to Index sizes
 */
template<typename Perm, typename IdxI, typename IdxJ, typename IdxK, typename IdxLin = int>
struct DLayout3d : public RAJA::Layout3d<Perm, IdxI, IdxJ, IdxK, IdxLin>{

  inline DLayout3d(Grid_Data *domain, int sdom_id) :
    RAJA::Layout3d<Perm, IdxI, IdxJ, IdxK, IdxLin>::Layout3d(
        domain->indexSize<IdxI>(sdom_id),
        domain->indexSize<IdxJ>(sdom_id),
        domain->indexSize<IdxK>(sdom_id))
  {}

  inline DLayout3d(int ni, int nj, int nk) :
    RAJA::Layout3d<Perm, IdxI, IdxJ, IdxK, IdxLin>::Layout3d(ni, nj, nk)
  {}

};


/**
 * Wrapper around Layout4d that provides accessors to Index sizes
 */
template<typename Perm, typename IdxI, typename IdxJ, typename IdxK, typename IdxL, typename IdxLin = int>
struct DLayout4d : public RAJA::Layout4d<Perm, IdxI, IdxJ, IdxK, IdxL, IdxLin>{

  inline DLayout4d(Grid_Data *domain, int sdom_id) :
    RAJA::Layout4d<Perm, IdxI, IdxJ, IdxK, IdxL, IdxLin>::Layout4d(
        domain->indexSize<IdxI>(sdom_id),
        domain->indexSize<IdxJ>(sdom_id),
        domain->indexSize<IdxK>(sdom_id),
        domain->indexSize<IdxL>(sdom_id))
  {}

  inline DLayout4d(int ni, int nj, int nk, int nl) :
    RAJA::Layout4d<Perm, IdxI, IdxJ, IdxK, IdxL, IdxLin>::Layout4d(ni, nj, nk, nl)
  {}

};

template<typename POL, typename IdxI, typename IdxJ, typename BODY>
void dForall2(Grid_Data *domain, int sdom_id, BODY const &body){

  RAJA::RangeSegment seg_i = domain->indexRange<IdxI>(sdom_id);
  RAJA::RangeSegment seg_j = domain->indexRange<IdxJ>(sdom_id);

  // Call underlying forall, extracting ranges from domain
  RAJA::forall2<POL, IdxI, IdxJ>(seg_i, seg_j, body);
}

template<typename POL, typename IdxI, typename IdxJ, typename IdxK, typename BODY>
void dForall3(Grid_Data *domain, int sdom_id, BODY const &body){

  RAJA::RangeSegment seg_i = domain->indexRange<IdxI>(sdom_id);
  RAJA::RangeSegment seg_j = domain->indexRange<IdxJ>(sdom_id);
  RAJA::RangeSegment seg_k = domain->indexRange<IdxK>(sdom_id);

  // Call underlying forall, extracting ranges from domain
  RAJA::forall3<POL, IdxI, IdxJ, IdxK>(seg_i, seg_j, seg_k, body);
}

template<typename POL, typename IdxI, typename IdxJ, typename IdxK, typename IdxL, typename BODY>
void dForall4(Grid_Data *domain, int sdom_id, BODY const &body){

  RAJA::RangeSegment seg_i = domain->indexRange<IdxI>(sdom_id);
  RAJA::RangeSegment seg_j = domain->indexRange<IdxJ>(sdom_id);
  RAJA::RangeSegment seg_k = domain->indexRange<IdxK>(sdom_id);
  RAJA::RangeSegment seg_l = domain->indexRange<IdxL>(sdom_id);

  // Call underlying forall, extracting ranges from domain
  RAJA::forall4<POL, IdxI, IdxJ, IdxK, IdxL>(seg_i, seg_j, seg_k, seg_l, body);
}

#endif



