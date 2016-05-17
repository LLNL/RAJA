/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */

/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining box segment classes.
 *     
 ******************************************************************************
 */

#ifndef RAJA_BoxSegment_HXX
#define RAJA_BoxSegment_HXX

#include "config.hxx"

#include "BaseSegment.hxx"

#include <algorithm>
#include <iosfwd>


namespace RAJA {


/*!
 ******************************************************************************
 *
 * \brief  Segment class representing a Box indices.
 *
 *         Box is specified by a corner, number of dimensions,
 *         extent in each dimentsion, and stride to advance in each dimension.
 *
 ******************************************************************************
 */
class BoxSegment : public BaseSegment
{
public:

   ///
   /// Default box segment ctor.
   ///
   /// Segment undefined until begin/end values set.
   ///
   BoxSegment()
   : BaseSegment( _BoxSeg_ ), 
     m_corner(UndefinedValue), 
     m_dim(UndefinedValue) { ; }

   ///
   /// Construct box segment with [begin, end) specified.
   ///
   BoxSegment(Index_type corner,  Index_type dim,
                Index_type const *extent, Index_type const *stride) 
   : BaseSegment( _BoxSeg_ ) 
   {
      m_corner = corner ;
      m_dim = dim ;

      for (int i=0; i<m_dim; ++i) {
         m_extent[i] = extent[i] ;
         m_stride[i] = stride[i] ;
      }

      /* eliminate trivially defined dimensions */
      for (int i = m_dim-1; i>0; --i) {
         if (m_extent[i] == 1) {
            if (i != m_dim-1) {
               for (int j=i; j<m_dim-1; ++j) {
                  m_extent[i] = m_extent[i+1] ;
                  m_stride[i] = m_stride[i+1] ;
               }
            }
            --m_dim ;
         }
      }

      /* collapse adjacent dimensions with regular stride */
      for (int i=0; i<(m_dim-1); ++i) {
         while (m_extent[i]*m_stride[i] == m_stride[i+1]) {
            m_extent[i] *= m_extent[i+1] ;
            for (int j=i+1; j<(m_dim-1); ++j) {
               m_extent[j] = m_extent[j+1] ;
               m_stride[j] = m_stride[j+1] ;
            }
            if (--m_dim == i+1) {
               break ;
            }
         }
      }

      for (int i=m_dim; i<3; ++i) {
         m_extent[i] = 0 ;
         m_stride[i] = 0 ;
      }
   }

   ///
   /// Destructor defined because some compilers don't appear to inline the
   /// one they generate.
   ///
   ~BoxSegment() {;}

   ///
   /// Copy ctor defined because some compilers don't appear to inline the
   /// one they generate.
   ///
   BoxSegment(const BoxSegment& other) 
   : BaseSegment( _BoxSeg_ ),
     m_corner(other.m_corner),
     m_dim(other.m_dim)
   {
      for (int i=0; i<3; ++i) {
         m_extent[i] = ((i<other.m_dim) ? other.m_extent[i] : 0);
         m_stride[i] = ((i<other.m_dim) ? other.m_stride[i] : 0);
      }
   }

   ///
   /// Copy assignment operator defined because some compilers don't 
   /// appear to inline the one they generate.
   ///
   BoxSegment& operator=(const BoxSegment& rhs)
   {
      if ( &rhs != this ) {
         BoxSegment copy(rhs);
         this->swap(copy);
      }
      return *this;
   }

   ///
   /// Swap function for copy-and-swap idiom.
   ///
   void swap(BoxSegment& other)
   {
      using std::swap;
      swap(m_corner, other.m_corner);
      swap(m_dim, other.m_dim);
      swap(m_extent[0], other.m_extent[0]) ;
      swap(m_extent[1], other.m_extent[1]) ;
      swap(m_extent[2], other.m_extent[2]) ;
      swap(m_stride[0], other.m_stride[0]) ;
      swap(m_stride[1], other.m_stride[1]) ;
      swap(m_stride[2], other.m_stride[2]) ;
   }


   ///
   /// Return starting offset for box. 
   ///
   Index_type getCorner() const { return m_corner; }

   ///
   /// Set starting offset for box. 
   ///
   void setCorner(Index_type corner) { m_corner = corner; }

   ///
   /// Return dimension of box. 
   ///
   Index_type getDim() const { return m_dim; }

   ///
   /// Set dimension of box.
   ///
   void setDim(Index_type dim) { m_dim = dim; }

   ///
   /// Return pointer to box extents. 
   ///
   Index_type const * getExtent() const { return &m_extent[0]; }

   ///
   /// Return pointer to box strides. 
   ///
   Index_type const * getStride() const { return &m_stride[0]; }

   ///
   /// Return number of indices represented by box.
   ///
   Index_type getLength() const { int prod = 1 ;
                                  for (int i=0; i<m_dim; ++i)
                                     prod *= m_extent[i] ;
                                  return (prod); }

   ///
   /// Return 'Owned' indicating that segment object owns the data
   /// representing its indices.
   ///
   IndexOwnership getIndexOwnership() const { return Owned; }

   ///
   /// Equality operator returns true if segments are equal; else false.
   ///
   bool operator ==(const BoxSegment& other) const
   {
      bool equal = ( (m_corner == other.m_corner) && (m_dim == other.m_dim) ) ;
      for (int i=0; i<m_dim; ++i) {
        equal = equal && (m_extent[i] == other.m_extent[i]);
        equal = equal && (m_stride[i] == other.m_stride[i]);
      }
      return equal ;
   }

   ///
   /// Inequality operator returns true if segments are not equal, else false.
   ///
   bool operator !=(const BoxSegment& other) const
   {
      return ( !(*this == other) );
   }

   ///
   /// Equality operator returns true if segments are equal; else false.
   /// (Implements pure virtual method in BaseSegment class).
   ///
   bool operator ==(const BaseSegment& other) const
   {
      const BoxSegment* o_ptr = dynamic_cast<const BoxSegment*>(&other);
      if ( o_ptr ) {
        return ( *this == *o_ptr );
      } else {
        return false;
      }
   }

   ///
   /// Inquality operator returns true if segments are not equal; else false.
   /// (Implements pure virtual method in BaseSegment class).
   ///
   bool operator !=(const BaseSegment& other) const
   {
      return ( !(*this == other) );
   }

   ///
   /// Print segment data to given output stream.
   ///
   void print(std::ostream& os) const;

private:
   Index_type m_corner; /* corner offset */
   Index_type m_dim;    /* number of box dimensions */
   Index_type m_extent[3] ;
   Index_type m_stride[3] ;
};


}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
