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

#include <vector> 
#include <iostream> 

//#define RAJA_USE_CTOR_DELEGATION
#undef RAJA_USE_CTOR_DELEGATION


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
   /// Construct unstructured index set from array of indices with given length.
   ///
   /// This operation performs deep copy of index array.
   ///
   UnstructuredISet(const Index_type* indx, Index_type len);

   ///
   /// Construct unstructured index set from std::vector of indices.
   ///
   /// This operation performs deep copy of vector data.
   ///
   UnstructuredISet(const std::vector<Index_type>& indx);

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
   void print(std::ostream& os) const;

private:
   //
   // Private default ctor used to simplify implementation of other
   // ctors and assignment operator. Class implementation relies on
   // C++11 ctor delegation. 
   //
   UnstructuredISet(Index_type len = 0);

   //
   // Allocate aligned index array.
   //  
   void allocateIndexData(Index_type len);

   Index_type* __restrict__ m_indx;
   Index_type  m_len;
};


}  // closing brace for namespace statement

#endif  // closing endif for header file include guard
