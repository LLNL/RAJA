/*
 * ArrayWrapper.hpp
 *
 *  Created on: Jul 30, 2016
 *      Author: rrsettgast
 */

#ifndef SRC_COMPONENTS_CORE_SRC_ARRAY_MULTIDIMENSIONALARRAY_HPP_
#define SRC_COMPONENTS_CORE_SRC_ARRAY_MULTIDIMENSIONALARRAY_HPP_
#include<vector>
#include<iostream>

#define ARRAY_BOUNDS_CHECK 0

#ifdef __clang__
#define restrict __restrict__
#define restrict_this
#elif __GNUC__
#define restrict __restrict__
#define restrict_this __restrict__
#endif


#if 0
#include "common/DataTypes.hpp"
#else
#include <assert.h>
using  integer_t     = int;
#endif


namespace multidimensionalArray
{

inline integer_t constexpr maxDim() { return 8; }

template < int N >
integer_t stride_helper( integer_t const * const restrict lengths )
{
  return lengths[0]*stride_helper<N-1>(lengths+1);
}
template<>
integer_t stride_helper<0>( integer_t const * const restrict )
{
  return 1;
}

template <int N>
integer_t stride( integer_t const * const restrict lengths )
{
	return stride_helper<N-1>(lengths+1);
}

template<int N>
integer_t * calcStrides( integer_t const * const restrict lengths )
{
  integer_t * strides = new integer_t[N];
  strides[N-1] = 1;
  for( integer_t a=N-2 ; a>=0 ; --a )
  {
    strides[a] = strides[a+1] * lengths[a+1];
  }

  return strides;
}





template< int COUNT, int JUNK, int ...REST >
struct UNPACK_HELPER
{
  inline static constexpr integer_t f()
  {
    return UNPACK_HELPER<COUNT-1,REST...>::f();
  }
};
template<int JUNK, int ...REST>
struct UNPACK_HELPER<0,JUNK,REST...>
{
  inline static constexpr integer_t f()
  {
    return JUNK;
  }
};




template< int COUNT, int JUNK, int ...REST >
struct STRIDE_HELPER
{
  inline static constexpr integer_t f()
  {
    return JUNK*STRIDE_HELPER<COUNT-1,REST...>::f();
  }
};

template< int JUNK, int ...REST >
struct STRIDE_HELPER<1,JUNK,REST...>
{
  inline static constexpr integer_t f()
  {
    return JUNK;
  }
};




template< int COUNT, int JUNK, int ...REST >
struct STRIDEFUNC
{
  inline static constexpr integer_t f()
  {
    return STRIDEFUNC<COUNT-1,REST...>::f();
  }
};

template< int JUNK, int ...REST >
struct STRIDEFUNC<0,JUNK,REST...>
{
  inline static constexpr integer_t f()
  {
    return STRIDE_HELPER<sizeof...(REST),REST...>::f();
  }
};




template< long long ...DIMENSIONS>
class dimensions
{
public:
  inline constexpr dimensions()
  {}

  inline static constexpr integer_t NDIMS() { return sizeof...(DIMENSIONS); }

  template< int DIM >
  inline static constexpr integer_t DIMENSION() { return UNPACK_HELPER<DIM,DIMENSIONS...>::f(); }

  template< int DIM >
  inline static constexpr
  typename std::enable_if<DIM!=sizeof...(DIMENSIONS)-1,integer_t>::type
  STRIDE() { return STRIDEFUNC<DIM,DIMENSIONS...>::f(); }

  template< int DIM >
  inline static constexpr
  typename std::enable_if<DIM==sizeof...(DIMENSIONS)-1,integer_t>::type
  STRIDE() { return 1; }


};


/**
 * @tparam T    type of data that is contained by the array
 * @tparam NDIM number of dimensions in array (e.g. NDIM=1->vector, NDIM=2->Matrix, etc. )
 * This class serves as a multidimensional array interface for a chunk of memory. This is a lightweight
 * class that contains only pointers, on integer data member to hold the stride, and another instantiation of type
 * ArrayAccesssor<T,NDIM-1> to represent the sub-array that is passed back upon use of operator[]. Pointers to the
 * data and length of each dimension passed into the constructor, thus the class does not own the data itself, nor does
 * it own the array that defines the shape of the data.
 */
template< typename T, int NDIM >
class ArrayAccessor
{
public:

  /// deleted default constructor
  ArrayAccessor() = delete;

  /**
   * @param data pointer to the beginning of the data
   * @param length pointer to the beginning of an array of lengths. This array has length NDIM
   *
   * Base constructor that takes in raw data pointers, sets member pointers, and calculates stride.
   */
  inline explicit constexpr ArrayAccessor( T * const restrict inputData,
                                           integer_t const * const restrict inputLength ):
    m_data(inputData),
    m_dims(inputLength),
    m_strides(calcStrides<NDIM>(inputLength))
  {}

  inline explicit constexpr ArrayAccessor( T * const restrict inputData,
                                           integer_t const * const restrict inputLength,
                                           integer_t const * const restrict inputStrides ):
    m_data(inputData),
    m_dims(inputLength),
    m_strides(inputStrides)
  {}

  /**
   * @param data pointer to the beginning of the data
   * @param length pointer to the beginning of an array of lengths. This array has length NDIM
   *
   * Base constructor that takes in raw data pointers, sets member pointers, and calculates stride.
   */
  inline explicit constexpr ArrayAccessor( T * const restrict inputData ):
    m_data(    inputData + maxDim() ),
    m_dims( reinterpret_cast<integer_t*>( inputData ) + 1 ),
    m_strides(calcStrides(m_dims))
  {}


  /// default destructor
  ~ArrayAccessor() = default;

  /**
   * @param source object to copy
   * copy constructor invokes direct copy of each member in source
   */
  inline constexpr ArrayAccessor( ArrayAccessor const & source ):
    m_data(source.m_data),
    m_dims(source.m_dims),
    m_strides(source.m_strides)
  {}

  /**
   * @param source object to move
   * move constructor invokes direct move of each member in source. In the case of data members that are pointers, this
   * is a straight copy. Not really a move, but rather a copy.
   */
  inline constexpr ArrayAccessor( ArrayAccessor && source ):
    m_data( std::move(source.m_data) ),
    m_dims( std::move(source.m_dims) ),
    m_strides(source.m_strides)
  {}
  
  /**
   * @param index index of the element in array to access
   * @return a reference to the member m_childInterface, which is of type ArrayAccessor<T,NDIM-1>.
   * This function sets the data pointer for m_childInterface.m_data to the location corresponding to the input
   * parameter "index". Thus, the returned object has m_data pointing to the beginning of the data associated with its
   * sub-array.
   */
  inline constexpr ArrayAccessor<T,NDIM-1> operator[](integer_t const index) const
  {
#if ARRAY_BOUNDS_CHECK == 1
    assert( index < m_dims[0] );
#endif
    return ArrayAccessor<T,NDIM-1>( &(m_data[ index*m_strides[0] ] ), m_dims+1, m_strides+1 );
  }

  template<int DIM = NDIM>
  inline constexpr T & operator()( integer_t const index0, integer_t const index1 ) const
  {
    return m_data[index0*m_strides[0]+index1];
  }

private:
  /// pointer to beginning of data for this array, or sub-array.
  T * const restrict m_data;

  /// pointer to array of length NDIM that contains the lengths of each array dimension
  integer_t const * const restrict m_dims;

  /// pointer to array of length NDIM that contains the lengths of each array dimension
  integer_t const * const restrict m_strides;


};




/**
 * Specialization for the ArrayAccessor<typename T, int NDIM> class template. This specialization defines the lowest
 * level ArrayAccessor in the array hierarchy. Thus this is the only level that actually allows data access, where the
 * other template instantiations only return references to a sub-array. In essence, this is a 1D array.
 */
template< typename T >
class ArrayAccessor<T,1>
{
public:

  /// deleted default constructor
  ArrayAccessor() = delete;

  /**
   * @param data pointer to the beginning of the data
   * @param length pointer to the beginning of an array of lengths. This array has length NDIM
   *
   * Base constructor that takes in raw data pointers, sets member pointers. Unlike the higher dimensionality arrays,
   * no calculation of stride is necessary for NDIM=1.
   */
  inline explicit constexpr ArrayAccessor( T * const restrict inputData,
                                           integer_t const * const restrict intputLength ):
    m_data(inputData),
    m_dims(intputLength),
    m_strides(1)

  {}

  inline explicit constexpr ArrayAccessor( T * const restrict inputData,
                                           integer_t const * const restrict inputLength,
                                           integer_t const * const restrict inputStrides ):
    m_data(inputData),
    m_dims(inputLength),
    m_strides(inputStrides)
  {}

  inline explicit constexpr ArrayAccessor( T * const restrict inputData ):
    m_data( static_cast<integer_t*>( inputData + 1 + static_cast<int>(2*(inputData[0])))),
    m_lengths( static_cast<integer_t*>( inputData + 1 ) ),
    m_strides(1)
  {}

  /// default destructor
  ~ArrayAccessor() = default;


  /**
   * @param source object to copy
   * copy constructor invokes direct copy of each member in source
   */
  ArrayAccessor( ArrayAccessor const & source ):
  m_data(source.m_data),
  m_dims(source.m_dims),
  m_strides(source.m_strides)
  {}

  /**
   * @param source object to move
   * move constructor invokes direct move of each member in source. In the case of data members that are pointers, this
   * is a straight copy. Not really a move, but rather a copy.
   */
  ArrayAccessor( ArrayAccessor && source ):
  m_data(source.m_data),
  m_dims(source.m_lengths),
  m_strides(source.m_strides)
  {}



  /**
   * @param index index of the element in array to access
   * @return a reference to the m_data[index], where m_data is a T*.
   * This function simply returns a reference to the pointer deferenced using index.
   */
  inline T & operator[](integer_t const index) const
  {
#if ARRAY_BOUNDS_CHECK == 1
    assert( index < m_dims[0] );
#endif
    return m_data[index];
  }

private:
  /// pointer to beginning of data for this array, or sub-array.
  T * const restrict m_data;

  /// pointer to array of length NDIM that contains the lengths of each array dimension
  integer_t const * const restrict m_dims;

  integer_t const * const restrict m_strides;
};




} /* namespace arraywrapper */

#endif /* SRC_COMPONENTS_CORE_SRC_ARRAY_MULTIDIMENSIONALARRAY_HPP_ */
