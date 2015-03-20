
/*!
 ******************************************************************************
 *
 * \brief  Special "segment" method that iterates over index set segments
 *         using "task-graph" omp parallel for execution policy.
 *
 *         This method assumes that a TaskGraphNode data structure has been
 *         properly set up for each segment in the index set.
 *
 ******************************************************************************
 */

#if 0

template <typename LOOP_BODY>
RAJA_INLINE
void forallSegments( const IndexSet& iss, LOOP_BODY loop_body )
{
   IndexSet &is = (*const_cast<IndexSet *>(&iss)) ;

   const int num_seg = is.getNumSegments();

#pragma omp parallel for schedule(static, 1)
   for ( int isi = 0; isi < num_seg; ++isi ) {
      volatile int *semMem =
         reinterpret_cast<volatile int *>(&is.segmentSemaphoreValue(isi)) ;

      while(*semMem != 0) {
         /* spin or (better) sleep here */ ;
        // printf("%d ", *semMem) ;
        // sleep(1) ;
        // volatile int spin ;
        // for (spin = 0; spin<1000; ++spin) {
        //    spin = spin ;
        // }
        sched_yield() ;
      }

      SegmentType segtype = is.getSegmentType(isi);
      const RangeISet* iset =
         static_cast<const RangeISet*>(is.getSegmentISet(isi));

      /* Produce a new indexset */
      IndexSet tmp;
      tmp.addRange(iset->getBegin(), iset->getEnd()) ;
      tmp.setPrivateData(0, is.getPrivateData(isi)) ;

      loop_body(&tmp) ;

      if (is.segmentSemaphoreReloadValue(isi) != 0) {
         is.segmentSemaphoreValue(isi) = is.segmentSemaphoreReloadValue(isi) ;
      }

      if (is.segmentSemaphoreNumDepTasks(isi) != 0) {
         for (int ii=0; ii<is.segmentSemaphoreNumDepTasks(isi); ++ii) {
           /* alternateively, we could get the return value of this call */
           /* and actively launch the task if we are the last depedent task. */
           /* in that case, we would not need the semaphore spin loop above */
           int seg = is.segmentSemaphoreDepTask(isi, ii) ;
           __sync_fetch_and_sub(&is.segmentSemaphoreValue(seg), 1) ;
         }
      }

   } // iterate over segments of hybrid index set
}

//#else  original alternative implementation...

// #define RAJA_USE_SCHED_LOOP 1

template <typename LOOP_BODY>
RAJA_INLINE
void forallSegments( const IndexSet& iss, LOOP_BODY loop_body )
{
  IndexSet *is = (const_cast<IndexSet *>(&iss)) ;

  const int num_seg = is->getNumSegments();

#pragma omp parallel
  {
    int numThreads = omp_get_max_threads() ;

#ifdef RAJA_USE_SCHED_LOOP
// guarantee thread binding matches static binding
#pragma omp for schedule(static, 1) nowait
    for (int tid = 0; tid<numThreads ;++tid) {
#else
    int tid = omp_get_thread_num() ;
#endif

      /* Create a temporary IndexSet with one Segment */
      IndexSet tmp;
      tmp.push_back( RangeSegment(0, 0) ) ; // create a dummy range

      for ( int isi = tid; isi < num_seg; isi += numThreads ) {

#if 0
        volatile int *semMem =
          reinterpret_cast<volatile int *>(&is->segmentSemaphoreValue(isi));

        while(*semMem != 0) {
          /* spin or (better) sleep here */ ;
          // printf("%d ", *semMem) ;
          // sleep(1) ;
          // volatile int spin ;
          // for (spin = 0; spin<1000; ++spin) {
          //    spin = spin ;
          // }
          sched_yield() ;
        }
#endif

        RangeSegment* isetSeg = static_cast<RangeSegment*>(is->getSegment(isi));

        segTmp->setBegin(isetSeg->getBegin()) ;
        segTmp->setEnd(isetSeg->getEnd()) ;
        segTmp->setPrivate(isetSeg->getPrivate()) ;

        loop_body(&tmp) ;

#if 0
        if (is->segmentSemaphoreReloadValue(isi) != 0) {
           is->segmentSemaphoreValue(isi) = is->segmentSemaphoreReloadValue(isi);
        }

        if (is->segmentSemaphoreNumDepTasks(isi) != 0) {
          for (int ii=0; ii<is->segmentSemaphoreNumDepTasks(isi); ++ii) {
            /* alternateively, we could get the return value of this call */
            /* and actively launch the task if we are the last depedent task. */
            /* in that case, we would not need the semaphore spin loop above */
            int seg = is->segmentSemaphoreDepTask(isi, ii) ;
            __sync_fetch_and_sub(&is->segmentSemaphoreValue(seg), 1) ;
          }
        }
#endif

      } // iterate over segments of hybrid index set
#ifdef RAJA_USE_SCHED_LOOP
    } // numThreads loop 
#endif
  }
}

#endif


template <typename LOOP_BODY>
RAJA_INLINE
void forallSegments( const IndexSet& iss, LOOP_BODY loop_body )
{
  IndexSet *is = (const_cast<IndexSet *>(&iss)) ;
  const int num_seg = is->getNumSegments();

#pragma omp parallel
  {
    int numThreads = omp_get_max_threads() ;
    int tid = omp_get_thread_num() ;

    /* Create a temporary IndexSet with one Segment */
    IndexSet is_tmp;
    is_tmp.push_back( RangeSegment(0, 0) ) ; // create a dummy range

    RangeSegment* segTmp = static_cast<RangeSegment*>(is_tmp.getSegment(0));

    for ( int isi = tid; isi < num_seg; isi += numThreads ) {

#if 0
      volatile int *semMem =
        reinterpret_cast<volatile int *>(&is->segmentSemaphoreValue(isi));

      while(*semMem != 0) {
        /* spin or (better) sleep here */ ;
        // printf("%d ", *semMem) ;
        // sleep(1) ;
        // volatile int spin ;
        // for (spin = 0; spin<1000; ++spin) {
        //    spin = spin ;
        // }
        sched_yield() ;
      }
#endif

      RangeSegment* isetSeg = static_cast<RangeSegment*>(is->getSegment(isi));

      segTmp->setBegin(isetSeg->getBegin()) ;
      segTmp->setEnd(isetSeg->getEnd()) ;
      segTmp->setPrivate(isetSeg->getPrivate()) ;

      loop_body(&is_tmp) ;

#if 0
      if (is->segmentSemaphoreReloadValue(isi) != 0) {
         is->segmentSemaphoreValue(isi) = is->segmentSemaphoreReloadValue(isi);
      }

      if (is->segmentSemaphoreNumDepTasks(isi) != 0) {
        for (int ii=0; ii<is->segmentSemaphoreNumDepTasks(isi); ++ii) {
          /* alternateively, we could get the return value of this call */
          /* and actively launch the task if we are the last depedent task. */
          /* in that case, we would not need the semaphore spin loop above */
          int seg = is->segmentSemaphoreDepTask(isi, ii) ;
          __sync_fetch_and_sub(&is->segmentSemaphoreValue(seg), 1) ;
        }
      }
#endif

    } // iterate over segments of hybrid index set
  }
}



