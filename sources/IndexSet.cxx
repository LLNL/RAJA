/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for index set classes
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#include "RAJA/IndexSet.hxx"

#include "RAJA/RangeSegment.hxx"
#include "RAJA/ListSegment.hxx"

#include <iostream>

namespace RAJA {


/*
*************************************************************************
*
* Public IndexSet methods.
*
*************************************************************************
*/

IndexSet::IndexSet()
: m_len(0)
{
}

IndexSet::IndexSet(const Index_type* const indices_in, Index_type length)
: m_len(0)
{
   buildIndexSet(*this, indices_in, length);
}

IndexSet::IndexSet(const IndexSet& other)
: m_len(0)
{
   copy(other); 
}

IndexSet& IndexSet::operator=(
   const IndexSet& rhs)
{
   if ( &rhs != this ) {
      IndexSet copy(rhs);
      this->swap(copy);
   }
   return *this;
}

IndexSet::~IndexSet()
{
   const int num_segs = getNumSegments();
   for ( int isi = 0; isi < num_segs; ++isi ) {
      SegmentType segtype = getSegmentType(isi);
      const void* iset = getSegmentISet(isi);

      if ( iset ) {

         switch ( segtype ) {

            case _RangeSeg_ : {
               RangeSegment* is =
                  const_cast<RangeSegment*>(
                     static_cast<const RangeSegment*>(iset)
                  );
               delete is;
               break;
            }

#if 0  // RDH RETHINK
            case _RangeStrideSeg_ : {
               RangeStrideSegment* is =
                  const_cast<RangeStrideSegment*>(
                     static_cast<const RangeStrideSegment*>(iset)
                  );
               delete is;
               break;
            }
#endif

            case _ListSeg_ : {
               ListSegment* is =
                  const_cast<ListSegment*>(
                     static_cast<const ListSegment*>(iset)
                  );
               delete is;
               break;
            }

            default : {
               std::cout << "\t IndexSet dtor: case not implemented!!\n";
            }

         }  // switch ( segtype )

      }  // if ( iset ) 

   }  // for isi...
}

void IndexSet::swap(IndexSet& other)
{
#if defined(RAJA_USE_STL)
   using std::swap;
   swap(m_len, other.m_len);
   swap(m_segments, other.m_segments);
#else
   m_len = other.m_len;
   m_segments = other.m_segments;
#endif
}


/*
*************************************************************************
*
* Methods to add segments to index set.
*
*************************************************************************
*/

void IndexSet::addRangeIndices(Index_type begin, Index_type end)
{
   RangeSegment* new_is = new RangeSegment(begin, end);
   addSegment( _RangeSeg_, new_is );
}

void IndexSet::addISet(const RangeSegment& iset)
{
   RangeSegment* new_is = new RangeSegment(iset);
   addSegment( _RangeSeg_, new_is );
}

#if 0  // RDH RETHINK
void IndexSet::addRangeStrideIndices(Index_type begin, Index_type end,
                                     Index_type stride)
{
   RangeStrideSegment* new_is = new RangeStrideSegment(begin, end, stride);
   addSegment( _RangeSeg_, new_is );
}

void IndexSet::addISet(const RangeStrideSegment& iset)
{
   RangeStrideSegment* new_is = new RangeStrideSegment(iset);
   addSegment( _RangeSeg_, new_is );
}
#endif

void IndexSet::addUnstructuredIndices(const Index_type* indx, 
                                      Index_type len,
                                      IndexOwnership indx_own)
{
   ListSegment* new_is = new ListSegment(indx, len, indx_own);
   addSegment( _ListSeg_, new_is );
}

void IndexSet::addISet(const ListSegment& iset, 
                         IndexOwnership indx_own)
{
   ListSegment* new_is = new ListSegment(iset.getIndex(),
                                         iset.getLength(),
                                         indx_own);
   addSegment( _ListSeg_, new_is );
}


/*
*************************************************************************
*
* Print contents of index set to given output stream.
*
*************************************************************************
*/

void IndexSet::print(std::ostream& os) const
{
   os << "INDEX SET : " 
      << getLength() << " length..." << std::endl
      << getNumSegments() << " segments..." << std::endl;

   const int num_segs = getNumSegments();
   for ( int isi = 0; isi < num_segs; ++isi ) {
      SegmentType segtype = getSegmentType(isi);
      const void* iset = getSegmentISet(isi);
      Index_type icount = getSegmentIcount(isi);

      os << "\nSegment " << isi << " : " << std::endl;

      switch ( segtype ) {

         case _RangeSeg_ : {
            if ( iset ) {
               os << "Icount = " << icount << std::endl; 
               const RangeSegment* is =
                  static_cast<const RangeSegment*>(iset);
               is->print(os);
            } else {
               os << "_RangeSeg_ is null" << std::endl;
            }
            break;
         }

#if 0  // RDH RETHINK
         case _RangeStrideSeg_ : {
            if ( iset ) {
               os << "Icount = " << icount << std::endl; 
               const RangeStrideSegment* is =
                  static_cast<const RangeStrideSegment*>(iset);
               is->print(os);
            } else {
               os << "_RangeStrideSeg_ is null" << std::endl;
            }
            break;
         }
#endif

         case _ListSeg_ : {
            if ( iset ) {
               os << "Icount = " << icount << std::endl; 
               const ListSegment* is =
                  static_cast<const ListSegment*>(iset);
               is->print(os);
            } else {
               os << "_ListSeg_ is null" << std::endl;
            }
            break;
         }

         default : {
            os << "IndexSet print: case not implemented!!\n";
         }

      }  // switch ( segtype )

   }  // for isi...
}


/*
*************************************************************************
*
* Private helper function to copy index set segments.
*
*************************************************************************
*/
void IndexSet::copy(const IndexSet& other)
{
   const int num_segs = other.getNumSegments();
   for ( int isi = 0; isi < num_segs; ++isi ) {
      SegmentType segtype = other.getSegmentType(isi);
      const void* iset = other.getSegmentISet(isi);

      if ( iset ) {

         switch ( segtype ) {

            case _RangeSeg_ : {
               addISet(*static_cast<const RangeSegment*>(iset));
               break;
            }

#if 0  // RDH RETHINK
            case _RangeStrideSeg_ : {
               addISet(*static_cast<const RangeStrideSegment*>(iset));
               break;
            }
#endif

            case _ListSeg_ : {
               addISet(*static_cast<const ListSegment*>(iset));
               break;
            }

            default : {
               std::cout << "\t IndexSet::copy: case not implemented!!\n";
            }

         }  // switch ( segtype )

      }  // if ( iset ) 

   }  // for isi...
}



/*
*************************************************************************
*
* IndexSet builder methods.
*
*************************************************************************
*/

void buildIndexSet(IndexSet& hiset,
                   const Index_type* const indices_in, 
                   Index_type length)
{
   if ( length == 0 ) return;

   /* only transform relatively large */
   if (length > RANGE_MIN_LENGTH) {
      /* build a rindex array from an index array */
      Index_type docount = 0 ;
      Index_type inrange = -1 ;

      /****************************/
      /* first, gather statistics */
      /****************************/

      Index_type scanVal = indices_in[0] ;
      Index_type sliceCount = 0 ;
      for (Index_type ii=1; ii<length; ++ii) {
         Index_type lookAhead = indices_in[ii] ;

         if (inrange == -1) {
            if ( (lookAhead == scanVal+1) && 
                 ((scanVal % RANGE_ALIGN) == 0) ) {
              inrange = 1 ;
            }
            else {
              inrange = 0 ;
            }
         }

         if (lookAhead == scanVal+1) {
            if ( (inrange == 0) && ((scanVal % RANGE_ALIGN) == 0) ) {
               if (sliceCount != 0) {
                  docount += 1 + sliceCount ; /* length + singletons */
               }
               inrange = 1 ;
               sliceCount = 0 ;
            }
            ++sliceCount ;  /* account for scanVal */
         }
         else {
            if (inrange == 1) {
               /* we can tighten this up by schleping any trailing */
               /* sigletons off into the subsequent singleton */
               /* array.  We would then also need to recheck the */
               /* final length of the range to make sure it meets */
               /* our minimum length crietria.  If it doesnt, */
               /* we need to emit a random array instead of */
               /* a range array */
               ++sliceCount ;
               docount += 2 ; /* length + begin */
               inrange = 0 ;
               sliceCount = 0 ;
            }
            else {
              ++sliceCount ;  /* account for scanVal */
            }
         }

         scanVal = lookAhead ;
      }  // end loop to gather statistics

      if (inrange != -1) {
         if (inrange) {
            ++sliceCount ;
            docount += 2 ; /* length + begin */
         }
         else {
            ++sliceCount ;
            docount += 1 + sliceCount ; /* length + singletons */
         }
      }
      else if (scanVal != -1) {
         ++sliceCount ;
         docount += 2 ;
      }
      ++docount ; /* zero length termination */

      /* What is the cutoff criteria for generating the rindex array? */
      if (docount < (length*(RANGE_ALIGN-1))/RANGE_ALIGN) {
         /* The rindex array can either contain a pointer into the */
         /* original index array, *or* it can repack the data from the */
         /* original index array.  Benefits of repacking could include */
         /* better use of hardware prefetch streams, or guaranteeing */
         /* alignment of index array segments. */

         /*******************************/
         /* now, build the rindex array */
         /*******************************/

         Index_type dobegin ;
         inrange = -1 ;

         scanVal = indices_in[0] ;
         sliceCount = 0 ;
         dobegin = scanVal ;
         for (Index_type ii=1; ii < length; ++ii) {
            Index_type lookAhead = indices_in[ii] ;

            if (inrange == -1) {
               if ( (lookAhead == scanVal+1) && 
                    ((scanVal % RANGE_ALIGN) == 0) ) {
                 inrange = 1 ;
               }
               else {
                 inrange = 0 ;
                 dobegin = ii-1 ;
               }
            }
            if (lookAhead == scanVal+1) {
               if ( (inrange == 0) && 
                    ((scanVal % RANGE_ALIGN) == 0) ) {
                  if (sliceCount != 0) {
                     hiset.addUnstructuredIndices(&indices_in[dobegin], 
                                                  sliceCount);
                  }
                  inrange = 1 ;
                  dobegin = scanVal ;
                  sliceCount = 0 ;
               }
               ++sliceCount ;  /* account for scanVal */
            }
            else {
               if (inrange == 1) {
               /* we can tighten this up by schleping any trailing */
               /* sigletons off into the subsequent singleton */
               /* array.  We would then also need to recheck the */
               /* final length of the range to make sure it meets */
               /* our minimum length crietria.  If it doesnt, */
               /* we need to emit a random array instead of */
               /* a range array */
                  ++sliceCount ;
                  hiset.addRangeIndices(dobegin, dobegin+sliceCount);
                  inrange = 0 ;
                  sliceCount = 0 ;
                  dobegin = ii ;
               }
               else {
                 ++sliceCount ;  /* account for scanVal */
               }
            }

            scanVal = lookAhead ;
         }  // for (Index_type ii ...

         if (inrange != -1) {
            if (inrange) {
               ++sliceCount ;
               hiset.addRangeIndices(dobegin, dobegin+sliceCount);
            }
            else {
               ++sliceCount ;
               hiset.addUnstructuredIndices(&indices_in[dobegin], sliceCount);
            }
         }
         else if (scanVal != -1) {
            hiset.addUnstructuredIndices(&scanVal, 1);
         }
      }
      else {  // !(docount < (length*RANGE_ALIGN-1))/RANGE_ALIGN)
         hiset.addUnstructuredIndices(indices_in, length);
      }
   }
   else {  // else !(length > RANGE_MIN_LENGTH)
      hiset.addUnstructuredIndices(indices_in, length);
   }
}

}  // closing brace for RAJA namespace
