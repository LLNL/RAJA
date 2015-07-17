/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Main Index Set API header file.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 *
 ******************************************************************************
 */

#ifndef IndexSet_HXX
#define IndexSet_HXX

#include <vector>
#include <iostream>
#include <cstdlib>

#define RAJA_SYNC 1

//////////////////////////////////////////////////////////////////////
//
//  FAULT TOLERANCE SUPPORT MACROS
//

#ifdef LULESH_FT

#ifdef REPORT_FT
#include "cycle.h"

#define FT_BEGIN \
   extern volatile int fault_type ; \
   bool repeat ; \
   bool do_time = false ; \
   ticks start = 0, stop = 0 ; \
   if ( fault_type != 0) { \
      printf("Uncaught fault %d\n", fault_type) ; \
      fault_type = 0 ; \
   } \
   do { \
      repeat = false ; \
      if (do_time) { \
         start = getticks() ; \
      }

#define FT_END \
      if (do_time) { \
         stop = getticks() ; \
         printf("recoverable fault time = %16f\n", elapsed(stop, start)/2.601e+9) ; \
         do_time = false ; \
         fault_type = 0 ; \
      } \
      if (fault_type < 0) { \
         printf("Unrecoverable fault (restart penalty)\n") ; \
         fault_type = 0 ; \
      } \
      if (fault_type > 0) { \
         /* invalidate cache */ \
         repeat = true ; \
         do_time = true ; \
      } \
   } while (repeat == true) ;

#else
#define FT_BEGIN \
   extern volatile int fault_type ; \
   bool repeat ; \
   if ( fault_type == 0) { \
      do { \
         repeat = false ;

#define FT_END \
         if (fault_type > 0) { \
            /* invalidate cache */ \
            repeat = true ; \
            fault_type = 0 ; \
         } \
      } while (repeat == true) ; \
   } \
   else { \
      fault_type = 0 ; /* ignore for the non-reporting simulation */ \
   }

#endif


#else

#define FT_BEGIN

#define FT_END

#endif

//
//  Variables for size of vector units
//
//     IndexSet_VECTOR_DOUBLE_ALIGN - value used in hybrid index set builder
//                                    methods to align structured ranges 
//                                    with vector units; units of "double"
//
//     IndexSet_RANGE_MIN_LENGTH - value used as min length of
//                                 structured range parts in hybrid
//                                 index set builder routines; units
//                                 of "double"
//
//     IndexSet_VECTOR_ALIGN - value passed to intrinsics to specify 
//                             alignment of data, loop bounds, etc.;
//                             units of "bytes" 


//
// Configuration for chaos platforms with AVX vector instructions.
//
const int IndexSet_VECTOR_DOUBLE_ALIGN = 4; // elements

const int IndexSet_RANGE_MIN_LENGTH = 32 ; // elements

const int IndexSet_VECTOR_ALIGN = 32; // bytes

//////////////////////////////////////////////////////////////////////
//
//  Variables for compiler instrinsics, directives, typedefs
//
//     INLINE_DIRECTIVE 
//
//     ALIGN_DATA(<variable>) - macro to express alignment of data,
//                              loop indices, etc.

//
// Configuration options for Intel compilers
//

#define INLINE_DIRECTIVE inline  __attribute__((always_inline))

#define ALIGN_DATA(d) 

typedef int    Index_type;


/*!
 ******************************************************************************
 *
 * \brief  Class representing a contiguous range of indices.
 *
 *         Range is specified by begin and end values.
 *         Traversal executes as:  
 *            for (i = m_begin; i < m_end; ++i) {
 *               expression using i as array index.
 *            }
 *
 ******************************************************************************
 */
class RangeIndexSet
{
public:

   RangeIndexSet(Index_type begin, Index_type end) 
     : m_begin(begin), m_end(end)
   { if (end-begin < IndexSet_RANGE_MIN_LENGTH ) /* abort */ ; }

   Index_type getBegin() const { return m_begin; }
   Index_type getEnd() const { return m_end; }

   Index_type getLength() const { return (m_end-m_begin); }

private:
   //
   // The default ctor is not implemented.
   //
   RangeIndexSet();

   Index_type m_begin;
   Index_type m_end;
};

/*!
 ******************************************************************************
 *
 * \brief  Class representing an arbitrary collection of indices. 
 *
 *         Length indicates number of indices in index array.
 *         Traversal executes as:  
 *            for (i = 0; i < m_len; ++i) {
 *               expression using m_indx[i] as array index.
 *            }
 *
 ******************************************************************************
 */
class UnstructuredIndexSet
{
public:

   UnstructuredIndexSet(Index_type begin, Index_type end) ;

   UnstructuredIndexSet(const Index_type* indx, Index_type len);

   UnstructuredIndexSet(const UnstructuredIndexSet& obj);

   UnstructuredIndexSet& operator=(const UnstructuredIndexSet& other);

   ~UnstructuredIndexSet();

   Index_type getLength() const { return m_len; }

   const Index_type* getIndex() const { return m_indx; }

private:
   //
   // The default ctor is not implemented.
   //
   UnstructuredIndexSet();

   //
   // Allocate new index array, copy indices to it and return it.
   //
   Index_type* copyIndices(const Index_type* indx, Index_type len);

   Index_type* __restrict__ m_indx;
   Index_type  m_len;
};


/*!
 ******************************************************************************
 *
 * \brief  Class representing an hybrid index set comprised of a collection
 *         of contiguous ranges and arbitrary indices.  
 *
 *         Each element in collection is called a "part" of the hybrid 
 *         index set.
 *
 ******************************************************************************
 */
class HybridIndexSet
{
public:

   ///
   /// Enum describing types of parts in hybrid index set and typedef
   /// to hold a single part
   ///
   enum IndexType { _Range_, _Unstructured_ };
   ///
   typedef std::pair<IndexType, const void*> PartPair;

   ///
   /// Construct empty hybrid index set
   ///
   HybridIndexSet();

   //
   // Copy-constructor for hybrid index set
   //
   HybridIndexSet(const HybridIndexSet&);

   //
   // Copy-assignment for hybrid index set
   //
   HybridIndexSet& operator=(const HybridIndexSet&);

   ///
   /// Hybrid index set destructor destroys all index set parts.
   ///
   ~HybridIndexSet();

   ///
   /// Create copy of given RangeIndexSet and add to hybrid index set.
   ///
   void addIndexSet(const RangeIndexSet& index_set);

   ///
   /// Add contiguous range of indices to hybrid index set as a RangeIndexSet.
   /// 
   void addRangeIndices(Index_type begin, Index_type end);

   ///
   /// Create copy of given UnstructuredIndexSet and add to hybrid index set.
   ///
   void addIndexSet(const UnstructuredIndexSet& index_set);

   ///
   /// Add array of indices to hybrid index set as an UnstructuredIndexSet. 
   /// 
   void addUnstructuredIndices(const Index_type* indx, Index_type len);

   ///
   /// Return total length of hybrid index set; i.e., sum of lengths
   /// over all parts.
   ///
   Index_type getLength() const { return m_len; }

   ///
   /// Return total length of hybrid index set; i.e., sum of lengths
   /// over all parts.
   ///
   Index_type getNumParts() const { return m_indexsets.size(); } 

   ///
   /// Return PartPair at given part index in hybrid index set.
   ///
   const PartPair& getPartPair(int part_isi) const 
      { return m_indexsets[part_isi]; }

private:
   //
   // Copy parts (deep copy) to given vector and return total length.
   //
   void copyParts(const HybridIndexSet& other);

   Index_type  m_len;
   std::vector<PartPair> m_indexsets;

}; 

/*!
 ******************************************************************************
 *
 * \brief  Tag structs for available index set traversal categories.
 *
 ******************************************************************************
 */

struct sequential_traversal {};

struct vectorized_traversal {};

struct omp_vectorized_traversal {};

struct cuda_traversal {};

/*!
 ******************************************************************************
 *
 * \brief  Traverse contiguous range of indices using aligned vectorization.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
INLINE_DIRECTIVE
void IndexSet_forall(sequential_traversal,
                     Index_type begin, Index_type end, LOOP_BODY loop_body)
{

   FT_BEGIN ;

//   __assume(begin % IndexSet_VECTOR_DOUBLE_ALIGN == 0) ;
#pragma loop_count min(IndexSet_RANGE_MIN_LENGTH)
#pragma novector
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii );
   }

   FT_END ;
}

template <typename LOOP_BODY>
INLINE_DIRECTIVE
void IndexSet_forall_min_loc(sequential_traversal,
                     Index_type begin, Index_type end,
                     double *min, int *loc,
                     LOOP_BODY loop_body)
{
   FT_BEGIN ;

//   __assume(begin % IndexSet_VECTOR_DOUBLE_ALIGN == 0) ;
#pragma loop_count min(IndexSet_RANGE_MIN_LENGTH)
#pragma novector
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, min, loc );
   }

   FT_END ;
}


template <typename LOOP_BODY>
__global__ void IndexSet_forall_kernel(LOOP_BODY loop_body, Index_type length)
{
  
  Index_type ii = blockDim.x * blockIdx.x + threadIdx.x;
  //printf("\n length = %d, ii = %d", (int)length, (int)ii);
  if (ii < length)
    loop_body(ii);
}


template <typename LOOP_BODY>
INLINE_DIRECTIVE
void IndexSet_forall(cuda_traversal,
                     Index_type begin, Index_type end, LOOP_BODY loop_body)
{

   FT_BEGIN ;

   //printf("\n XXX: IndexSet_forall:: cuda_traversal, length = %d",
	//  (int) (end - begin));
   
   size_t blockSize = 256;
   size_t gridSize = (end - begin) / blockSize + 1;
   IndexSet_forall_kernel<<<gridSize, blockSize>>>(loop_body, end - begin);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) { 
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
		<< __LINE__ << std::endl;
      exit(1);
   }
#endif

   FT_END ;
}


template <typename LOOP_BODY>
__global__ void IndexSet_forall_min_loc_kernel(LOOP_BODY loop_body, 
					       double *min, int *loc,
					       Index_type length)
{
  Index_type ii = blockDim.x * blockIdx.x + threadIdx.x;
  if (ii < length)
    loop_body(ii, min, loc);
  
}

template <typename LOOP_BODY>
INLINE_DIRECTIVE
void IndexSet_forall_min_loc(cuda_traversal,
                     Index_type begin, Index_type end,
                     double *min, int *loc,
                     LOOP_BODY loop_body)
{
   FT_BEGIN ;
    
   size_t blockSize = 256;
   size_t gridSize = (end - begin) / blockSize + 1;
   IndexSet_forall_min_loc_kernel<<<gridSize, blockSize>>>(loop_body, min, loc,
						   end - begin);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) { 
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
		<< __LINE__ << std::endl;
      exit(1);
   }
#endif
   
   FT_END ;
}



template <typename LOOP_BODY>
INLINE_DIRECTIVE
void IndexSet_forall(vectorized_traversal,
                     Index_type begin, Index_type end, LOOP_BODY loop_body)
{

   FT_BEGIN ;

//   __assume(begin % IndexSet_VECTOR_DOUBLE_ALIGN == 0) ;
#pragma loop_count min(IndexSet_RANGE_MIN_LENGTH)
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii );
   }

   FT_END ;
}

template <typename LOOP_BODY>
INLINE_DIRECTIVE
void IndexSet_forall_min_loc(vectorized_traversal,
                     Index_type begin, Index_type end,
                     double *min, int *loc,
                     LOOP_BODY loop_body)
{
   FT_BEGIN ;

//   __assume(begin % IndexSet_VECTOR_DOUBLE_ALIGN == 0) ;
#pragma loop_count min(IndexSet_RANGE_MIN_LENGTH)
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, min, loc );
   }

   FT_END ;
}

template <typename LOOP_BODY>
INLINE_DIRECTIVE
void IndexSet_forall(omp_vectorized_traversal,
                     Index_type begin, Index_type end, LOOP_BODY loop_body)
{
   FT_BEGIN ;

//   __assume(begin % IndexSet_VECTOR_DOUBLE_ALIGN == 0) ;
#pragma loop_count min(IndexSet_RANGE_MIN_LENGTH)
#pragma omp parallel for
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii );
   }

   FT_END ;
}

template <typename LOOP_BODY>
INLINE_DIRECTIVE
void IndexSet_forall_min_loc(omp_vectorized_traversal,
                     Index_type begin, Index_type end,
                     double *min, int *loc,
                     LOOP_BODY loop_body)
{
#if defined(_OPENMP)
   Index_type threads = omp_get_max_threads();

   /* align these temps to coherence boundaries? */
   double  minTmp[threads];
   Index_type locTmp[threads];

   FT_BEGIN ;

   for (Index_type i=0; i<threads; ++i) {
       minTmp[i] = *min ;
       locTmp[i] = *loc ;
   }

#pragma omp parallel
   {
      int myThread = omp_get_thread_num() ;
     
//      __assume(begin % IndexSet_VECTOR_DOUBLE_ALIGN == 0) ;
#pragma loop_count min(IndexSet_RANGE_MIN_LENGTH)
#pragma omp for
      for ( Index_type ii = begin ; ii < end ; ++ii ) {
         loop_body( ii, &minTmp[myThread], &locTmp[myThread] );
      }
   }

   for (Index_type i = 1; i < threads; ++i) {
      if (minTmp[i] < minTmp[0] ) {
         minTmp[0]    = minTmp[i];
         locTmp[0] = locTmp[i];
      }
   }

   FT_END ;

   *min = minTmp[0] ;
   *loc = locTmp[0] ;
#endif /* defined(_OPENMP) */
}

/*!
 ******************************************************************************
 *
 * \brief  Traverse unstructured index set.  This is here as a convenience.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
INLINE_DIRECTIVE
void IndexSet_forall(sequential_traversal,
                     const UnstructuredIndexSet& is, LOOP_BODY loop_body)
{
   FT_BEGIN ;

   const Index_type len = is.getLength();
   const Index_type* __restrict__ idx = is.getIndex();
   ALIGN_DATA(idx);
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k] );
   }

   FT_END ;
}

template <typename LOOP_BODY>
INLINE_DIRECTIVE
void IndexSet_forall_min_loc(sequential_traversal,
                     const UnstructuredIndexSet& is,
                     double *min, int *loc,
                     LOOP_BODY loop_body)
{
   FT_BEGIN ;

   const Index_type len = is.getLength();
   const Index_type* __restrict__ idx = is.getIndex();
   ALIGN_DATA(idx);
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], min, loc );
   }

   FT_END ;
}



template <typename LOOP_BODY>
__global__ void IndexSet_forall_unstructured_kernel
	(LOOP_BODY loop_body, const Index_type *idx, Index_type length)
{
  Index_type ii = blockDim.x * blockIdx.x + threadIdx.x;
  if (ii < length)
    loop_body(idx[ii]);
}


template <typename LOOP_BODY>
INLINE_DIRECTIVE
void IndexSet_forall(cuda_traversal,
                     const UnstructuredIndexSet& is, LOOP_BODY loop_body)
{
   FT_BEGIN ;

   const Index_type len = is.getLength();
   const Index_type *idx = is.getIndex();
   ALIGN_DATA(idx);
   
   
   size_t blockSize = 256;
   size_t gridSize = len / blockSize + 1;
   IndexSet_forall_unstructured_kernel<<<gridSize, blockSize>>>(loop_body, idx, len);
   
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) { 
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
		<< __LINE__ << std::endl;
      exit(1);
   }
#endif

   FT_END ;
}


template <typename LOOP_BODY>
__global__ void IndexSet_forall_min_loc_unstructured_kernel
	(LOOP_BODY loop_body, const Index_type *idx, double *min, int *loc,
	 Index_type length)
{
  Index_type ii = blockDim.x * blockIdx.x + threadIdx.x;
  if (ii < length)
    loop_body(idx[ii], min, loc);
}


template <typename LOOP_BODY>
INLINE_DIRECTIVE
void IndexSet_forall_min_loc(cuda_traversal,
                     const UnstructuredIndexSet& is,
                     double *min, int *loc,
                     LOOP_BODY loop_body)
{
   FT_BEGIN ;

   const Index_type len = is.getLength();
   const Index_type* __restrict__ idx = is.getIndex();
   ALIGN_DATA(idx);
   
   size_t blockSize = 256;
   size_t gridSize = len / blockSize + 1;
   IndexSet_forall_min_loc_unstructured_kernel<<<gridSize, blockSize>>>(loop_body, idx, min, loc, len);
   
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) { 
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
		<< __LINE__ << std::endl;
      exit(1);
   }
#endif
   
   FT_END ;
}


template <typename LOOP_BODY>
INLINE_DIRECTIVE
void IndexSet_forall(vectorized_traversal,
                     const UnstructuredIndexSet& is, LOOP_BODY loop_body)
{
   /* indirect access is never vectorized by the compiler */
   IndexSet_forall<sequential_traversal>(is, loop_body) ;
}

template <typename LOOP_BODY>
INLINE_DIRECTIVE
void IndexSet_forall_min_loc(vectorized_traversal,
                     const UnstructuredIndexSet& is,
                     double *min, int *loc,
                     LOOP_BODY loop_body)
{
   /* indirect access is never vectorized by the compiler */
   IndexSet_forall_min_loc<sequential_traversal>(is, min, loc, loop_body) ;
}

template <typename LOOP_BODY>
INLINE_DIRECTIVE
void IndexSet_forall(omp_vectorized_traversal,
                     const UnstructuredIndexSet& is, LOOP_BODY loop_body)
{
   FT_BEGIN ;

   const Index_type len = is.getLength();
   const Index_type* __restrict__ idx = is.getIndex();
   ALIGN_DATA(idx);
#pragma omp parallel for
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k] );
   }

   FT_END ;
}

template <typename LOOP_BODY>
INLINE_DIRECTIVE
void IndexSet_forall_min_loc(omp_vectorized_traversal,
                     const UnstructuredIndexSet& is,
                     double *min, int *loc,
                     LOOP_BODY loop_body)
{
#if defined(_OPENMP)
   Index_type threads = omp_get_max_threads();

   /* align these temps to coherence boundaries? */
   double  minTmp[threads];
   Index_type locTmp[threads];

   FT_BEGIN ;

   for (Index_type i=0; i<threads; ++i) {
       minTmp[i] = *min ;
       locTmp[i] = *loc ;
   }

#pragma omp parallel
   {
      int myThread = omp_get_thread_num() ;

      const Index_type len = is.getLength();
      const Index_type* __restrict__ idx = is.getIndex();
      ALIGN_DATA(idx);
#pragma omp for
      for ( Index_type k = 0 ; k < len ; ++k ) {
         loop_body( idx[k], &minTmp[myThread], &locTmp[myThread] );
      }
   }

   for (Index_type i = 1; i < threads; ++i) {
      if (minTmp[i] < minTmp[0] ) {
         minTmp[0]    = minTmp[i];
         locTmp[0] = locTmp[i];
      }
   }

   FT_END ;

   *min = minTmp[0] ;
   *loc = locTmp[0] ;
#endif /* defined(_OPENMP) */
}



//////////////////////////////////////////////////////////////////////
//
// Function templates to traverse variants of range index set
// with no index set object.
//
//////////////////////////////////////////////////////////////////////

/*!
 ******************************************************************************
 *
 * \brief Traversal of index set using template parameter to specify traversal. 
 *
 ******************************************************************************
 */
template <typename TRAVERSE_T,
          typename LOOP_BODY>
INLINE_DIRECTIVE
void IndexSet_forall(Index_type begin, Index_type end, LOOP_BODY loop_body)
{
   IndexSet_forall(TRAVERSE_T(),
                   begin, end, loop_body);
}

template <typename TRAVERSE_T,
          typename LOOP_BODY>
INLINE_DIRECTIVE
void IndexSet_forall_min_loc(Index_type begin, Index_type end,
                     double *min, int *loc,  LOOP_BODY loop_body)
{
   IndexSet_forall_min_loc(TRAVERSE_T(),
                           begin, end, min, loc, loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief  Traverse RangeIndexSet using traversal method specified by
 *         template parameter.
 *
 ******************************************************************************
 */
template <typename TRAVERSE_T, typename LOOP_BODY>
INLINE_DIRECTIVE
void IndexSet_forall(TRAVERSE_T traversal,
                     const RangeIndexSet& is, LOOP_BODY loop_body)
{
   IndexSet_forall( traversal,
                    is.getBegin(), is.getEnd(), loop_body );
}

template <typename TRAVERSE_T, typename LOOP_BODY>
INLINE_DIRECTIVE
void IndexSet_forall_min_loc(TRAVERSE_T traversal,
                     const RangeIndexSet& is,
                     double *min, int *loc,
                     LOOP_BODY loop_body)
{
   IndexSet_forall_min_loc( traversal,
                    is.getBegin(), is.getEnd(),
                    min, loc, loop_body );
}


/*!
 ******************************************************************************
 *
 * \brief  Traverse HybridIndexSet using traversal method specified by
 *         template parameter on all parts.
 *
 ******************************************************************************
 */
template <typename TRAVERSE_T, typename LOOP_BODY>
void IndexSet_forall(TRAVERSE_T traversal,
                     const HybridIndexSet& is, LOOP_BODY loop_body)
{
   for ( int isi = 0; isi < is.getNumParts(); ++isi ) {
      const HybridIndexSet::PartPair& is_pair = is.getPartPair(isi);

      switch ( is_pair.first ) {

         case HybridIndexSet::_Range_ : {
            IndexSet_forall(traversal,
               *(static_cast<const RangeIndexSet*>(is_pair.second)),
               loop_body
            );
            break;
         }

         case HybridIndexSet::_Unstructured_ : {
            IndexSet_forall(traversal,
               *(static_cast<const UnstructuredIndexSet*>(is_pair.second)),
               loop_body
            );
            break;
         }

         default : {
         }

      }  // switch ( is_pair.first ) {

   } // iterate over parts of hybrid index set
}

template <typename TRAVERSE_T, typename LOOP_BODY>
void IndexSet_forall_min_loc(TRAVERSE_T traversal,
                     const HybridIndexSet& is,
                     double *min, int *loc,
                     LOOP_BODY loop_body)
{
   for ( int isi = 0; isi < is.getNumParts(); ++isi ) {
      const HybridIndexSet::PartPair& is_pair = is.getPartPair(isi);

      switch ( is_pair.first ) {

         case HybridIndexSet::_Range_ : {
            IndexSet_forall_min_loc(traversal,
               *(static_cast<const RangeIndexSet*>(is_pair.second)),
               min, loc, loop_body
            );
            break;
         }

         case HybridIndexSet::_Unstructured_ : {
            IndexSet_forall_min_loc(traversal,
               *(static_cast<const UnstructuredIndexSet*>(is_pair.second)),
               min, loc, loop_body
            );
            break;
         }

         default : {
         }
      }  // switch ( is_pair.first ) {
   } // iterate over parts of hybrid index set
}

/*!
 ******************************************************************************
 *
 * \brief Traversal of index set using template parameter to specify traversal. 
 *
 ******************************************************************************
 */
template <typename TRAVERSE_T,
          typename INDEXSET_T, typename LOOP_BODY>
INLINE_DIRECTIVE
void IndexSet_forall(const INDEXSET_T& is, LOOP_BODY loop_body)
{
   IndexSet_forall(TRAVERSE_T(), is, loop_body);
}

template <typename TRAVERSE_T,
          typename INDEXSET_T, typename LOOP_BODY>
INLINE_DIRECTIVE
void IndexSet_forall_min_loc(const INDEXSET_T& is, double *min, int *loc,
                     LOOP_BODY loop_body)
{
   IndexSet_forall_min_loc(TRAVERSE_T(), is, min, loc, loop_body);
}


/*!
 ******************************************************************************
 *
 * \brief Set entries in given std::vector<Index_type> object to indices in 
 *        given RangeIndexSet that satisfy given condition.
 *
 *        Routine does no error-checking on argements and assumes size of
 *        Index_type vector is sufficient to hold all indices.
 *
 ******************************************************************************
 */
template <typename CONDITIONAL>
void IndexSet_getIndices(std::vector<Index_type>& indices,
                         const RangeIndexSet& is,
                         CONDITIONAL conditional)
{
   const Index_type begin = is.getBegin();
   const Index_type end = is.getEnd();

   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      if ( conditional( ii ) ) indices.push_back(ii); 
   }
}


/*!
 ******************************************************************************
 *
 * \brief Set entries in given std::vector<Index_type> object to indices in 
 *        given UnstructuredIndexSet that satisfy given condition.
 *
 *        Routine does no error-checking on argements and assumes size of
 *        Index_type vector is sufficient to hold all indices.
 *
 ******************************************************************************
 */
template <typename CONDITIONAL>
void IndexSet_getIndices(std::vector<Index_type>& indices,
                         const UnstructuredIndexSet& is,
                         CONDITIONAL conditional)
{
   const Index_type len = is.getLength();
   const Index_type* __restrict__ idx = is.getIndex();

   for ( Index_type k = 0 ; k < len ; ++k ) {
      if ( conditional( idx[k] ) ) indices.push_back( idx[k] );
   }
}

/*!
 ******************************************************************************
 *
 * \brief Set entries in given std::vector<Index_type> object to indices in 
 *        given HybridIndexSet that satisfy given condition.
 *
 *        Routine does no error-checking on argements and assumes size of
 *        Index_type vector is sufficient to hold all indices.
 *
 ******************************************************************************
 */
template <typename CONDITIONAL>
void IndexSet_getIndices(std::vector<Index_type>& indices,
                         const HybridIndexSet& is,
                         CONDITIONAL conditional)
{
   for ( Index_type isi = 0; isi < is.getNumParts(); ++isi ) {
      const HybridIndexSet::PartPair& is_pair = is.getPartPair(isi);

      switch ( is_pair.first ) {

         case HybridIndexSet::_Range_ : {
            IndexSet_getIndices(
               indices,
               *(static_cast<const RangeIndexSet*>(is_pair.second)),
               conditional 
            );
            break;
         }

         case HybridIndexSet::_Unstructured_ : {
            IndexSet_getIndices(
               indices,
               *(static_cast<const UnstructuredIndexSet*>(is_pair.second)),
               conditional
            );
            break;
         }

         default : {
         }

      }  // switch ( is_pair.first ) {

   } // iterate over parts of hybrid index set
}

#endif  // closing endif for header file include guard
