/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining unstructured index set classes.
 *     
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_UnstructuredISet_HXX
#define RAJA_UnstructuredISet_HXX

#include "config.hxx"

#include "datatypes.hxx"

#include "execpolicy.hxx"

#include <iostream> 


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
class UnstructuredISet
{
public:

   ///
   /// Sequential execution policy for unstructured index set.
   ///
   typedef RAJA::seq_exec seq_policy;

   ///
   /// Construct unstructured index set from given array with specified 
   /// length.
   ///
   /// By default the ctor performs deep copy of array elements.
   /// If false is passed as last argument, the constructed object
   /// does not own the index data and will hold a pointer to given data.
   ///
   UnstructuredISet(const Index_type* indx, Index_type len,
                    bool owns_index = true);

   ///
   /// Construct unstructured index set from arbitrary object holding 
   /// indices using a deep copy of given data.
   ///
   /// The object must have the following methods: size(), begin(), end().
   ///
   template< typename T>
   UnstructuredISet(const T& indx);

   ///
   /// Copy-constructor for unstructured index set
   ///
   UnstructuredISet(const UnstructuredISet& obj);

   ///
   /// Copy-assignment for unstructured index set
   ///
   UnstructuredISet& operator=(const UnstructuredISet& other);

   ///
   /// Destroy index set included its contents.
   ///
   ~UnstructuredISet();

   ///
   /// Swap function for copy-and-swap idiom.
   ///
   void swap(UnstructuredISet& other);

   ///
   ///  Return number of indices in index set.
   ///
   Index_type getLength() const { return m_len; }

   ///
   ///  Return const pointer to array of indices in index set.
   ///
   const Index_type* getIndex() const { return m_indx; }

   ///
   /// Return boolean indicating whether index set owns the data
   /// representing its indices.
   ///
   bool ownsIndex() const { return m_owns_index; }
    
   ///
   /// Print index set data, including segments, to given output stream.
   ///
   void print(std::ostream& os) const;

private:
   //
   // Initialize index data properly based on whether index set object
   // owns the index data.
   //  
   void initIndexData(const Index_type* indx, Index_type len,
                      bool owns_index);

   Index_type* __restrict__ m_indx;
   Index_type  m_len;
   bool m_owns_index;
};


/*!
 *  Implementation of generic constructor template.
 */ 
template< typename T> 
UnstructuredISet::UnstructuredISet(const T& indx)
: m_indx(0), m_len(0), m_owns_index(false)
{
   if ( indx.size() > 0 ) {
      m_len = indx.size();
      m_owns_index = true;
      m_indx = new Index_type[m_len];
      std::copy(indx.begin(), indx.end(), m_indx);
   } 
}


}  // closing brace for namespace statement

#endif  // closing endif for header file include guard
