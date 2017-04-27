#ifndef RAJA_rangelist_forall_HPP
#define RAJA_rangelist_forall_HPP

namespace RAJA {
namespace impl {

/*!
 ******************************************************************************
 *
 * \brief Execute Range or List segment from forall traversal method.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T, typename LOOP_BODY>
RAJA_INLINE void executeRangeList_forall(const IndexSetSegInfo* seg_info,
                                         LOOP_BODY&& loop_body)
{
  const BaseSegment* iseg = seg_info->getSegment();
  SegmentType segtype = iseg->getType();

  switch (segtype) {
    case _RangeSeg_: {
      const RangeSegment* tseg = static_cast<const RangeSegment*>(iseg);
      impl::forall(SEG_EXEC_POLICY_T(), *tseg, loop_body);
      break;
    }

#if 0  // RDH RETHINK
    case _RangeStrideSeg_ : {
         const RangeStrideSegment* tseg =
            static_cast<const RangeStrideSegment*>(iseg);
         impl::forall(
            SEG_EXEC_POLICY_T(),
            tseg->getBegin(), tseg->getEnd(), tseg->getStride(),
            loop_body
         );
         break;
      }
#endif

    case _ListSeg_: {
      const ListSegment* tseg = static_cast<const ListSegment*>(iseg);
      impl::forall(SEG_EXEC_POLICY_T(), *tseg, loop_body);
      break;
    }

    default: {
    }

  }  // switch on segment type
}

/*!
 ******************************************************************************
 *
 * \brief Execute Range or List segment from forall_Icount traversal method.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T, typename LOOP_BODY>
RAJA_INLINE void executeRangeList_forall_Icount(const IndexSetSegInfo* seg_info,
                                                LOOP_BODY&& loop_body)
{
  const BaseSegment* iseg = seg_info->getSegment();
  SegmentType segtype = iseg->getType();

  Index_type icount = seg_info->getIcount();

  switch (segtype) {
    case _RangeSeg_: {
      const RangeSegment* tseg = static_cast<const RangeSegment*>(iseg);
      impl::forall_Icount(SEG_EXEC_POLICY_T(), *tseg, icount, loop_body);
      break;
    }

#if 0  // RDH RETHINK
    case _RangeStrideSeg_ : {
         const RangeStrideSegment* tseg =
            static_cast<const RangeStrideSegment*>(iseg);
         forall_Icount(
            SEG_EXEC_POLICY_T(),
            tseg->getBegin(), tseg->getEnd(), tseg->getStride(),
            icount,
            loop_body
         );
         break;
      }
#endif

    case _ListSeg_: {
      const ListSegment* tseg = static_cast<const ListSegment*>(iseg);
      impl::forall_Icount(SEG_EXEC_POLICY_T(), *tseg, icount, loop_body);
      break;
    }

    default: {
    }

  }  // switch on segment type
}

/*!
 ******************************************************************************
 *
 * \brief  Generic wrapper for IndexSet policies to allow the use of normal
 * policies with them.
 *
 ******************************************************************************
 */
// TODO: this should be with the IndexSet class, really it should be part of
// its built-in iterator, but we need to address the include snarl first
template<typename SEG_EXEC_POLICY_T, typename LOOP_BODY>
struct rangeListExecutor {
  constexpr rangeListExecutor(LOOP_BODY &&body) : body(body) {}
  RAJA_INLINE
  void operator()(const IndexSetSegInfo &seg_info) {
    executeRangeList_forall<SEG_EXEC_POLICY_T>(&seg_info, body);
  }

 private:
  // LOOP_BODY body;
  typename std::remove_reference<LOOP_BODY>::type body;
};

template<typename SEG_EXEC_POLICY_T, typename LOOP_BODY>
constexpr RAJA_INLINE rangeListExecutor<SEG_EXEC_POLICY_T, LOOP_BODY>
makeRangeListExecutor(LOOP_BODY &&body) {
  return rangeListExecutor<SEG_EXEC_POLICY_T, LOOP_BODY>(body);
}

template<typename SEG_IT_POLICY_T,
    typename SEG_EXEC_POLICY_T,
    typename LOOP_BODY>
RAJA_INLINE void forall(
    IndexSet::ExecPolicy <SEG_IT_POLICY_T, SEG_EXEC_POLICY_T>,
    const IndexSet &iset,
    LOOP_BODY loop_body) {
  impl::forall(SEG_IT_POLICY_T(),
               iset, makeRangeListExecutor<SEG_EXEC_POLICY_T>(loop_body));
}

template<typename SEG_EXEC_POLICY_T, typename LOOP_BODY>
struct rangeListIcountExecutor {
  constexpr rangeListIcountExecutor(LOOP_BODY &&body) : body(body) {}
  RAJA_INLINE
  void operator()(const IndexSetSegInfo &seg_info) {
    executeRangeList_forall_Icount<SEG_EXEC_POLICY_T>(&seg_info, body);
  }

 private:
  typename std::remove_reference<LOOP_BODY>::type body;
  // LOOP_BODY body;
};

template<typename SEG_EXEC_POLICY_T, typename LOOP_BODY>
constexpr RAJA_INLINE rangeListIcountExecutor<SEG_EXEC_POLICY_T, LOOP_BODY>
makeRangeListIcountExecutor(LOOP_BODY &&body) {
  return rangeListIcountExecutor<SEG_EXEC_POLICY_T, LOOP_BODY>(body);
}

template<typename SEG_IT_POLICY_T,
    typename SEG_EXEC_POLICY_T,
    typename LOOP_BODY>
RAJA_INLINE void forall_Icount(
    IndexSet::ExecPolicy <SEG_IT_POLICY_T, SEG_EXEC_POLICY_T>,
    const IndexSet &iset,
    LOOP_BODY loop_body) {
  // no need for icount variant here
  impl::forall(SEG_IT_POLICY_T(),
               iset, makeRangeListIcountExecutor<SEG_EXEC_POLICY_T>(loop_body));
}

} // end of namespace impl
} // end of namespace RAJA

#endif // RAJA_rangelist_forall_HPP
