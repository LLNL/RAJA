// This work was performed under the auspices of the U.S. Department of Energy by
// Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344.?

/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file for simple vector template class that enables 
 *          RAJA to be used with or without the C++ STL.
 *     
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_RAJAVec_HXX
#define RAJA_RAJAVec_HXX

#include <algorithm>


namespace RAJA {

/*!
 ******************************************************************************
 *
 * \brief  Class template that provides a simple vector implementation 
 *         sufficient to insulate RAJA entities from the STL.
 *
 *         Note: This class has limited functionality sufficient to 
 *               support its usage for RAJA IndexSet operations. However,
 *               it does provide a push_front method that is not found
 *               in the STL vector container.
 *  
 *               Template type should support standard semantics for
 *               copy, swap, etc. 
 *
 ******************************************************************************
 */
template< typename T>
class RAJAVec
{
public:
   //
   // Construct empty vector with given capacity.
   //
   explicit RAJAVec(unsigned init_cap = 0)
   : m_capacity(0), m_size(0), m_data(0)
   {
      grow_cap(init_cap);
   }

   //
   // Copy ctor for vector.
   //
   RAJAVec(const RAJAVec<T>& other) 
   : m_capacity(0), m_size(0), m_data(0)
   {
      copy(other); 
   }

   ///
   /// Swap function for copy-and-swap idiom.
   /// 
   void swap(RAJAVec<T>& other)
   {
      using std::swap;
      swap(m_capacity, other.m_capacity);
      swap(m_size, other.m_size);
      swap(m_data, other.m_data);
   }

   //
   // Copy-assignment operator for vector.
   //
   RAJAVec<T>& operator=(const RAJAVec<T>& rhs) 
   {
      if ( &rhs != this ) {
         RAJAVec<T> copy(rhs);
         this->swap(copy);
      }
      return *this;
   }

   //
   // Destroy vector and its data.
   //
   ~RAJAVec()
   {
      if (m_capacity > 0) delete [] m_data;  
   }

   //
   // Return true if vector has size zero; false otherwise.
   //
   unsigned empty() const {
      return (m_size == 0) ;
   }

   //
   // Return current size of vector.
   //
   unsigned size() const { 
      return m_size;
   }

   //
   // Const bracket operator.
   //
   const T& operator [] (unsigned i) const
   {
      return m_data[i];
   }

   //
   // Non-const bracket operator.
   //
   T& operator [] (unsigned i)
   {
      return m_data[i];
   }

   //
   // Add item to back end of vector.
   //
   void push_back(const T& item) 
   {
      push_back_private(item);
   }

   //
   // Add item to front end of vector.
   //
   void push_front(const T& item)
   {
      push_front_private(item);
   }


private:
   //
   // Copy function for copy-and-swap idiom (deep copy).
   //
   void copy(const RAJAVec<T>& other)
   {
      grow_cap(other.m_capacity);
      for (unsigned i = 0; i < other.m_size; ++i) {
         m_data[i] = other[i];
      }
      m_capacity = other.m_capacity;
      m_size = other.m_size;
   }


   //
   // The following private members and methods provide a quick and dirty 
   // memory allocation scheme to mimick std::vector behavior without 
   // relying on STL directly.  These are initialized at the end of this file.
   //
   static const unsigned s_init_cap;
   static const double   s_grow_fac;

   unsigned nextCap(unsigned current_cap) 
   {
      if (current_cap == 0) { return s_init_cap; }
      return static_cast<unsigned>( current_cap * s_grow_fac ); 
   }

   void grow_cap(unsigned target_size)
   {
      unsigned target_cap = m_capacity;
      while ( target_cap < target_size ) { target_cap = nextCap(target_cap); } 

      if ( m_capacity < target_cap ) {
         T* tdata = new T[target_cap];

         if ( m_data ) {
            for (unsigned i = 0; (i < m_size)&&(i < target_cap); ++i) {
               tdata[i] = m_data[i];
            } 
            delete[] m_data;
         }

         m_data = tdata;
         m_capacity = target_cap;
      }
   }

   void push_back_private(const T& item)
   {
      grow_cap(m_size+1);
      m_data[m_size] = item;
      m_size++;
   }


   void push_front_private(const T& item)
   {
      size_t old_size = m_size;
      grow_cap( old_size+1 );

      for (unsigned i = old_size; i > 0; --i) {
         m_data[i] = m_data[i-1];
      }
      m_data[0] = item;
      m_size++;
   }
   

   unsigned m_capacity;
   unsigned m_size;
   T* m_data;
};

/*
*************************************************************************
*
* Initialize static members
*
*************************************************************************
*/
template< typename T>
const unsigned RAJAVec<T>::s_init_cap = 8;
template< typename T>
const double   RAJAVec<T>::s_grow_fac = 1.5;

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
