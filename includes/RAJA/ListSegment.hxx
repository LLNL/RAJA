/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining list segment classes.
 *     
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_ListSegment_HXX
#define RAJA_ListSegment_HXX

#include "BaseSegment.hxx"

#include "execpolicy.hxx"

#if defined(RAJA_USE_STL)
#include <utility> 
#include <algorithm> 
#endif

#include <iosfwd> 



namespace RAJA {


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
class ListSegment : public BaseSegment
{
public:

   ///
   /// Sequential execution policy for list segment.
   ///
   typedef RAJA::seq_exec seq_policy;

   ///
   /// Construct list segment from given array with specified length.
   ///
   /// By default the ctor performs deep copy of array elements.
   /// If 'Unowned' is passed as last argument, the constructed object
   /// does not own the segment data and will hold a pointer to given data.
   /// In this case, caller must manage object lifetimes properly.
   ///
   ListSegment(const Index_type* indx, Index_type len,
               IndexOwnership indx_own = Owned);

#if defined(RAJA_USE_STL)
   ///
   /// Construct list segment from arbitrary object holding 
   /// indices using a deep copy of given data.
   ///
   /// The object must provide methods: empty(), begin(), end().
   ///
   template< typename T> explicit ListSegment(const T& indx);
#endif

   ///
   /// Copy-constructor for list segment.
   ///
   ListSegment(const ListSegment& obj);

   ///
   /// Copy-assignment for list segment.
   ///
   ListSegment& operator=(const ListSegment& other);

   ///
   /// Destroy segment including its contents.
   ///
   ~ListSegment();

   ///
   /// Swap function for copy-and-swap idiom.
   ///
   void swap(ListSegment& other);

   ///
   ///  Return number of indices in segment.
   ///
   Index_type getLength() const { return m_len; }

   ///
   ///  Return const pointer to array of indices in segment.
   ///
   const Index_type* getIndex() const { return m_indx; }

   ///
   /// Return enum value indicating whether segment object owns the data
   /// representing its indices.
   ///
   IndexOwnership getIndexOwnership() const { return m_indx_own; }
    
   ///
   /// Print segment data to given output stream.
   ///
   void print(std::ostream& os) const;

private:
   //
   // The default ctor is not implemented.
   //
   ListSegment();

   //
   // Initialize segment data properly based on whether object
   // owns the index data.
   //  
   void initIndexData(const Index_type* indx, Index_type len,
                      IndexOwnership indx_own);

   Index_type* __restrict__ m_indx;
   Index_type  m_len;
   IndexOwnership m_indx_own;
};


#if defined(RAJA_USE_STL)
/*!
 ******************************************************************************
 *
 *  \brief Implementation of generic constructor template.
 *
 ******************************************************************************
 */ 
template< typename T> 
ListSegment::ListSegment(const T& indx)
: BaseSegment( _ListSeg_ ),
  m_indx(0), m_len(0), m_indx_own(Unowned)
{
   if ( !indx.empty() ) {
      m_len = indx.size();
      m_indx = new Index_type[m_len];
      std::copy(indx.begin(), indx.end(), m_indx);
      m_indx_own = Owned;
   } 
}
#endif


}  // closing brace for RAJA namespace 


#if defined(RAJA_USE_STL)
/*!
 *  Specialization of std swap method.
 */ 
namespace std {

template< > 
RAJA_INLINE
void swap(RAJA::ListSegment& a, RAJA::ListSegment& b)
{
   a.swap(b);
}

}
#endif


#endif  // closing endif for header file include guard
