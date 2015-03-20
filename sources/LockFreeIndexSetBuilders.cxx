/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for lock-free index set builder methods.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#include "RAJA/IndexSet.hxx"

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <iostream>

namespace RAJA {

/*
 * See buildLockFreeIndexSet.hxx for other comments.
 */

/*
 ******************************************************************************
 *
 * Build Lock-free "block" index set (planar division).
 *
 * Note: Method assumes IndexSet ptr refers to an empty index set.
 *
 ******************************************************************************
 */
#define PROFITABLE_ENTITY_THRESHOLD_BLOCK 100

void buildLockFreeBlockIndexset(RAJA::IndexSet& iset,
                                int fastDim, int midDim, int slowDim)
{
   int numThreads = omp_get_max_threads() ;

   // printf("Lock-free created\n") ;

   if ((midDim | slowDim) == 0)  /* 1d mesh */
   {
      if (fastDim/PROFITABLE_ENTITY_THRESHOLD_BLOCK <= 1) {
          // printf("%d %d\n", 0, fastDim) ;
          iset.push_back( RAJA::RangeSegment(0, fastDim) ) ;
      }
      else {
         /* This just sets up the schedule -- a truly safe */
         /* execution of this schedule would require a check */
         /* for completion of dependent threads before execution. */

         /* We might want to force one thread if the */
         /* profitability ratio is really bad, but for */
         /* now use the brain dead approach. */
         int numSegments = numThreads*3 ;
         for (int lane=0; lane<3; ++lane) {
            for (int i=lane; i<numSegments; i += 3)
            {
               int start = i*fastDim/numSegments ;
               int end   = (i+1)*fastDim/numSegments ;
               // printf("%d %d\n", start, end) ;
               iset.push_back( RAJA::RangeSegment(start, end) ) ;
            }
         }
      }
   }
   else if (slowDim == 0) /* 2d mesh */
   {
      int rowsPerSegment = midDim/(3*numThreads) ;
      if (rowsPerSegment == 0) {
          // printf("%d %d\n", 0, fastDim*midDim) ;
          iset.push_back( RAJA::RangeSegment(0, fastDim*midDim) ) ;
      }
      else {
         /* This just sets up the schedule -- a truly safe */
         /* execution of this schedule would require a check */
         /* for completion of dependent threads before execution. */

         /* We might want to force one thread if the */
         /* profitability ratio is really bad, but for */
         /* now use the brain dead approach. */
         int numSegments = 3*numThreads ;
         int segmentSize = midDim / numThreads ;
         int segmentThresh = midDim % numSegments ;
         for (int lane=0; lane<3; ++lane) {
            for (int i=0; i<numThreads; ++i)
            {
               int startRow = i*midDim/numThreads ;
               int endRow = (i+1)*midDim/numThreads ;
               int start = startRow*fastDim ;
               int end   = endRow*fastDim ;
               int len = end - start ;
               // printf("%d %d\n", start + (lane  )*len/3,
               //                   start + (lane+1)*len/3  ) ;
               iset.push_back( RAJA::RangeSegment(start + (lane  )*len/3,
                                                  start + (lane+1)*len/3) ) ;
            }
         }
      }
   }
   else { /* 3d mesh */
      /* Need at least 3 full planes per thread */
      /* and at least one segment per plane */
      const int segmentsPerThread = 2 ;
      int rowsPerSegment = slowDim/(segmentsPerThread*numThreads) ;
      if (rowsPerSegment == 0) {
          // printf("%d %d\n", 0, fastDim*midDim*slowDim) ;
          iset.push_back( RAJA::RangeSegment(0, fastDim*midDim*slowDim) ) ;
          printf("Failure to create lockfree indexset\n") ;
          exit(-1) ;
      }
      else {
         /* This just sets up the schedule -- a truly safe */
         /* execution of this schedule would require a check */
         /* for completion of dependent threads before execution. */

         /* We might want to force one thread if the */
         /* profitability ratio is really bad, but for */
         /* now use the brain dead approach. */
         int numSegments = segmentsPerThread*numThreads ;
         int segmentSize = slowDim / numThreads ;
         int segmentThresh = slowDim % numSegments ;
         for (int lane=0; lane<segmentsPerThread; ++lane) {
            for (int i=0; i<numThreads; ++i)
            {
               int startPlane = i*slowDim/numThreads ;
               int endPlane = (i+1)*slowDim/numThreads ;
               int start = startPlane*fastDim*midDim ;
               int end   = endPlane*fastDim*midDim ;
               int len = end - start ;
               // printf("%d %d\n", start + (lane  )*len/segmentsPerThread,
               //                   start + (lane+1)*len/segmentsPerThread  );
               iset.push_back( RAJA::RangeSegment(
                                     start + (lane  )*len/segmentsPerThread,
                                     start + (lane+1)*len/segmentsPerThread) );
            }
         }

         /* Allocate dependency graph structures for index set segments */
         iset.initDependencyGraph();

         if (segmentsPerThread == 1) {
            /* This dependency graph should impose serialization */
            for (int i=0; i<numThreads; ++i) {
               RAJA::DepGraphNode* task  = 
                  iset.getSegmentInfo(i)->getDepGraphNode();
               task->semaphoreValue() = (( i == 0) ? 0 : 1) ;
               task->semaphoreReloadValue() = ((i == 0) ? 0 : 1) ;
               if (i != numThreads-1) {
                  task->numDepTasks() = 1 ;
                  task->depTaskNum(0) = i+1 ;
               }
            }
         }
         else {
            /* This dependency graph relies on omp schedule(static, 1) */
            /* but allows a minimumal set of dependent tasks be used */
            int borderSeg = numThreads*(segmentsPerThread-1) ;
            for (int i=1; i<numThreads; ++i) {
               RAJA::DepGraphNode* task  = 
                  iset.getSegmentInfo(i)->getDepGraphNode();
               task->semaphoreReloadValue() = 1 ;
               task->numDepTasks() = 1 ;
               task->depTaskNum(0) = borderSeg + i - 1 ;

               RAJA::DepGraphNode* border_task  = 
                  iset.getSegmentInfo(borderSeg+i-1)->getDepGraphNode();
               border_task->semaphoreValue() = 1 ;
               border_task->semaphoreReloadValue() = 1 ;
               border_task->numDepTasks() = 1 ;
               border_task->depTaskNum(0) = i ;
            }
         }
      }
   }

   /* Print the dependency schedule for segments */
// iset.print(std::cout);

}


/*
 ******************************************************************************
 *
 * Build Lock-free "color" index set. The domain-set is colored based on
 * connectivity to the range-set.  All elements in each segment are
 * independent, and no two segments can be executed in parallel.
 *
 * Note: Method assumes IndexSet ptr refers to an empty index set.
 *
 ******************************************************************************
 */
void buildLockFreeColorIndexset(RAJA::IndexSet& iset,
                                int *domainToRange, int numEntity,
                                int numRangePerDomain, int numEntityRange,
                                int *elemPermutation,
                                int *ielemPermutation)
{
   bool done = false ;
   bool* isMarked = new bool[numEntity];

   int numWorkset = 0 ;
   int* worksetDelim = new int[numEntity];

   int worksetSize = 0 ;
   int* workset = new int[numEntity];

   int* rangeToDomain = new int[numEntityRange*numRangePerDomain] ;
   int* rangeToDomainCount = new int[numEntityRange] ;

   memset(rangeToDomainCount, 0, numEntityRange*sizeof(int)) ;

   /* create an inverse mapping */
   for (int i=0; i<numEntity; ++i) {
      for (int j=0 ; j<numRangePerDomain; ++j) {
         int id = domainToRange[i*numRangePerDomain+j] ;
         int idx = id*numRangePerDomain + rangeToDomainCount[id]++ ;
         if(idx > numEntityRange*numRangePerDomain ||
            rangeToDomainCount[id] > numRangePerDomain) {
            printf("foiled!\n") ;
            exit(-1) ;
         }
         rangeToDomain[idx] = i ;
      }
   }

   while (!done) {
      done = true ;

      for (int i=0; i<numEntity; ++i) {
         isMarked[i] = false ;
      }

      for (int i=0; i<worksetSize; ++i) {
         isMarked[workset[i]] = true ;
      }

      for (int i=0; i<numEntity; ++i) {

         if (isMarked[i] == false) {
            done = false ;
            if (worksetSize >= numEntity) {
               printf("foiled!\n") ;
               exit(-1) ;
            }
            workset[worksetSize++] = i ;
            for (int j=0 ; j<numRangePerDomain; ++j) {
               int id = domainToRange[i*numRangePerDomain+j] ;
               for (int k=0; k<rangeToDomainCount[id]; ++k) {
                  int idx = rangeToDomain[id*numRangePerDomain+k] ;
                  if (idx < 0 || idx >= numEntity) {
                     printf("foiled!\n") ;
                     exit(-1) ;
                  }
                  isMarked[idx] = true ;
               }
            }
         }
      }
      if (done == false) {
         worksetDelim[numWorkset++] = worksetSize ;
      }
   }

   delete [] rangeToDomainCount ;
   delete [] rangeToDomain ;

   if (worksetSize != numEntity) {
      printf("foiled!!!\n") ;
      exit (-1) ;
   }

   /* we may want to create a permutation array here */
   if (elemPermutation != 0l) {
      /* send back permutaion array, and corresponding range segments */

      memcpy(elemPermutation, &workset[0], numEntity*sizeof(int)) ;
      if (ielemPermutation != 0l) {
         for (int i=0; i<numEntity; ++i) {
            ielemPermutation[elemPermutation[i]] = i ;
         }
      }
      int end = 0 ;
      for (int i=0; i<numWorkset; ++i) {
         int begin = end ;
         end = worksetDelim[i] ;
         iset.push_back( RAJA::RangeSegment(begin, end) ) ;
      }
   }
   else {
      int end = 0 ;
      for (int i=0; i<numWorkset; ++i) {
         int begin = end ;
         end = worksetDelim[i] ;
         bool isRange = true ;
         for (int j=begin+1; j<end; ++j) {
            if (workset[j-1]+1 != workset[j]) {
               isRange = false ;
               break ;
            }
         }
         if (isRange) {
            iset.push_back( RAJA::RangeSegment( workset[begin], 
                                                  workset[end-1]+1) ) ;
         }
         else {
            iset.push_back( RAJA::ListSegment( &workset[begin], end-begin) );
            // printf("segment %d\n", i) ;
            // for (int j=begin; j<end; ++j) {
            //    printf("%d\n", workset[j]) ;
            // }
         }
      }
   }

   delete [] isMarked ;
   delete [] worksetDelim ;
   delete [] workset ;
}

}  // closing brace for RAJA namespace


