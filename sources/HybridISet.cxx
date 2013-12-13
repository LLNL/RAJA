/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for hybrid index set classes
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#include "RAJA/HybridISet.hxx"

#include <cstdlib>
#include <iostream>


namespace RAJA {


/*
*************************************************************************
*
* Public HybridISet methods.
*
*************************************************************************
*/

HybridISet::HybridISet()
: m_len(0)
{
}

HybridISet::HybridISet(const HybridISet& other)
: m_len(0)
{
   copy(other); 
}

HybridISet& HybridISet::operator=(
   const HybridISet& rhs)
{
   if ( &rhs != this ) {
      HybridISet copy(rhs);
      this->swap(copy);
   }
   return *this;
}

HybridISet::~HybridISet()
{
   const int num_segs = getNumSegments();
   for ( int isi = 0; isi < num_segs; ++isi ) {
      SegmentType segtype = getSegmentType(isi);
      const void* iset = getSegmentISet(isi);

      if ( iset ) {

         switch ( segtype ) {

            case _Range_ : {
               RangeISet* is =
                  const_cast<RangeISet*>(
                     static_cast<const RangeISet*>(iset)
                  );
               delete is;
               break;
            }

#if 0  // RDH RETHINK
            case _RangeStride_ : {
               RangeStrideISet* is =
                  const_cast<RangeStrideISet*>(
                     static_cast<const RangeStrideISet*>(iset)
                  );
               delete is;
               break;
            }
#endif

            case _Unstructured_ : {
               UnstructuredISet* is =
                  const_cast<UnstructuredISet*>(
                     static_cast<const UnstructuredISet*>(iset)
                  );
               delete is;
               break;
            }

            default : {
               std::cout << "\t HybridISet dtor: case not implemented!!\n";
            }

         }  // switch ( segtype )

      }  // if ( iset ) 

   }  // for isi...
}

void HybridISet::swap(HybridISet& other)
{
   using std::swap;
   swap(m_len, other.m_len);
   swap(m_segments, other.m_segments);
}


/*
*************************************************************************
*
* Methods to add segments to hybrid index set.
*
*************************************************************************
*/

void HybridISet::addRangeIndices(Index_type begin, Index_type end)
{
   RangeISet* new_is = new RangeISet(begin, end);
   m_segments.push_back(Segment( _Range_, new_is ));
   m_len += new_is->getLength();
}

void HybridISet::addISet(const RangeISet& iset)
{
   RangeISet* new_is = new RangeISet(iset);
   m_segments.push_back(Segment( _Range_, new_is ));
   m_len += new_is->getLength();
}

#if 0  // RDH RETHINK
void HybridISet::addRangeStrideIndices(Index_type begin, Index_type end,
                                       Index_type stride)
{
   RangeStrideISet* new_is = new RangeStrideISet(begin, end, stride);
   m_segments.push_back( Segment( _RangeStride_, new_is ));
   m_len += new_is->getLength();
}

void HybridISet::addISet(const RangeStrideISet& iset)
{
   RangeStrideISet* new_is = new RangeStrideISet(iset);
   m_segments.push_back(Segment( _RangeStride_, new_is, ));
   m_len += new_is->getLength();
}
#endif

void HybridISet::addUnstructuredIndices(const Index_type* indx, 
                                        Index_type len,
                                        IndexOwnership indx_own)
{
   UnstructuredISet* new_is = new UnstructuredISet(indx, len, indx_own);
   m_segments.push_back(Segment( _Unstructured_, new_is ));
   m_len += new_is->getLength();
}

void HybridISet::addISet(const UnstructuredISet& iset, 
                         IndexOwnership indx_own)
{
   UnstructuredISet* new_is = new UnstructuredISet(iset.getIndex(),
                                                   iset.getLength(),
                                                   indx_own);
   m_segments.push_back(Segment( _Unstructured_, new_is ));
   m_len += new_is->getLength();
}


/*
*************************************************************************
*
* Print contents of hybrid index set to given output stream.
*
*************************************************************************
*/

void HybridISet::print(std::ostream& os) const
{
   os << "HYBRID INDEX SET : " 
      << getLength() << " length..." << std::endl
      << getNumSegments() << " segments..." << std::endl;

   const int num_segs = getNumSegments();
   for ( int isi = 0; isi < num_segs; ++isi ) {
      SegmentType segtype = getSegmentType(isi);
      const void* iset = getSegmentISet(isi);

      os << "\tSegment " << isi << " : " << std::endl;

      switch ( segtype ) {

         case _Range_ : {
            if ( iset ) {
               const RangeISet* is =
                  static_cast<const RangeISet*>(iset);
               is->print(os);
            } else {
               os << "_Range_ is null" << std::endl;
            }
            break;
         }

#if 0  // RDH RETHINK
         case _RangeStride_ : {
            if ( iset ) {
               const RangeStrideISet* is =
                  static_cast<const RangeStrideISet*>(iset);
               is->print(os);
            } else {
               os << "_RangeStride_ is null" << std::endl;
            }
            break;
         }
#endif

         case _Unstructured_ : {
            if ( iset ) {
               const UnstructuredISet* is =
                  static_cast<const UnstructuredISet*>(iset);
               is->print(os);
            } else {
               os << "_Unstructured_ is null" << std::endl;
            }
            break;
         }

         default : {
            os << "HybridISet print: case not implemented!!\n";
         }

      }  // switch ( segtype )

   }  // for isi...
}


/*
*************************************************************************
*
* Private helper function to copy hybrid index set segments.
*
*************************************************************************
*/
void HybridISet::copy(const HybridISet& other)
{
   const int num_segs = getNumSegments();
   for ( int isi = 0; isi < num_segs; ++isi ) {
      SegmentType segtype = getSegmentType(isi);
      const void* iset = getSegmentISet(isi);

      if ( iset ) {

         switch ( segtype ) {

            case _Range_ : {
               addISet(*static_cast<const RangeISet*>(iset));
               break;
            }

#if 0  // RDH RETHINK
            case _RangeStride_ : {
               addISet(*static_cast<const RangeStrideISet*>(iset));
               break;
            }
#endif

            case _Unstructured_ : {
               addISet(*static_cast<const UnstructuredISet*>(iset));
               break;
            }

            default : {
               std::cout << "\t HybridISet::copy: case not implemented!!\n";
            }

         }  // switch ( segtype )

      }  // if ( iset ) 

   }  // for isi...
}



/*
*************************************************************************
*
* HybridISet builder methods.
*
*************************************************************************
*/

HybridISet* buildHybridISet(const Index_type* const indices_in, 
                            Index_type length)
{
   HybridISet* hindex = new HybridISet();

   /*
    * If index array information is suspect, return empty index set. 
    */
   if ( !indices_in || length == 0 ) return( hindex );

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
                     hindex->addUnstructuredIndices(&indices_in[dobegin], sliceCount);
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
                  hindex->addRangeIndices(dobegin, dobegin+sliceCount);
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
               hindex->addRangeIndices(dobegin, dobegin+sliceCount);
            }
            else {
               ++sliceCount ;
               hindex->addUnstructuredIndices(&indices_in[dobegin], sliceCount);
            }
         }
         else if (scanVal != -1) {
            hindex->addUnstructuredIndices(&scanVal, 1);
         }
      }
      else {  // !(docount < (length*RANGE_ALIGN-1))/RANGE_ALIGN)
         hindex->addUnstructuredIndices(indices_in, length);
      }
   }
   else {  // else !(length > RANGE_MIN_LENGTH)
      hindex->addUnstructuredIndices(indices_in, length);
   }

   return( hindex );
}

}  // closing brace for RAJA namespace
