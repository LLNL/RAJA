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
      const BaseSegment* seg = getSegment(isi);

      if ( seg ) {
         delete seg;
      } 

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

void IndexSet::push_back(const RangeSegment& segment)
{
   RangeSegment* new_seg = new RangeSegment(segment);
   push_back_private( new_seg );
}

void IndexSet::push_front(const RangeSegment& segment)
{
   RangeSegment* new_seg = new RangeSegment(segment);
   push_front_private( new_seg );
}


#if 0  // RDH RETHINK
void IndexSet::push_back(const RangeStrideSegment& segment)
{
   RangeStrideSegment* new_seg = new RangeSegment(segment);
   push_back_private( new_seg );
}

void IndexSet::push_front(const RangeStrideSegment& segment)
{
   RangeStrideSegment* new_seg = new RangeSegment(segment);
   push_front_private( new_seg );
}
#endif

void IndexSet::push_back(const ListSegment& iset, 
                                 IndexOwnership indx_own)
{
   ListSegment* new_seg = new ListSegment(iset.getIndex(),
                                          iset.getLength(),
                                          indx_own);
   push_back_private( new_seg );
}

void IndexSet::push_front(const ListSegment& iset,
                                  IndexOwnership indx_own)
{
   ListSegment* new_seg = new ListSegment(iset.getIndex(),
                                          iset.getLength(),
                                          indx_own);
   push_front_private( new_seg );
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

   for ( int isi = 0; isi < m_segments.size(); ++isi ) {

      const BaseSegment* iseg = getSegment(isi);
      SegmentType segtype = iseg->getType();

      os << "\nSegment " << isi << " : " << std::endl;

      switch ( segtype ) {

         case _RangeSeg_ : {
            if ( iseg ) {
               static_cast<const RangeSegment*>(iseg)->print(os);
            } else {
               os << "_RangeSeg_ is null" << std::endl;
            }
            break;
         }

#if 0  // RDH RETHINK
         case _RangeStrideSeg_ : {
            if ( iseg ) {
               static_cast<const RangeStrideSegment*>(iseg)->print(os);
            } else {
               os << "_RangeStrideSeg_ is null" << std::endl;
            }
            break;
         }
#endif

         case _ListSeg_ : {
            if ( iseg ) {
               static_cast<const ListSegment*>(iseg)->print(os);
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
* Note: Assumes this index set is empty.
*
*************************************************************************
*/
void IndexSet::copy(const IndexSet& other)
{
   const int num_segs = other.getNumSegments();
   for ( int isi = 0; isi < num_segs; ++isi ) {

      const BaseSegment* iseg = other.getSegment(isi);
      SegmentType segtype = iseg->getType();

      if ( iseg ) {

         switch ( segtype ) {

            case _RangeSeg_ : {
               push_back(*static_cast<const RangeSegment*>(iseg));
               break;
            }

#if 0  // RDH RETHINK
            case _RangeStrideSeg_ : {
               push_back(*static_cast<const RangeStrideSegment*>(iseg));
               break;
            }
#endif

            case _ListSeg_ : {
               push_back(*static_cast<const ListSegment*>(iseg));
               break;
            }

            default : {
               std::cout << "\t IndexSet::copy: case not implemented!!\n";
            }

         }  // switch ( segtype )

      }  // if ( iset ) 

   }  // for isi...
}


}  // closing brace for RAJA namespace
