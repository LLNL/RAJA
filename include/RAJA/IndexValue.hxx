/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */

#ifndef RAJA_INDEXVALUE_HXX__
#define RAJA_INDEXVALUE_HXX__

#include<RAJA/int_datatypes.hxx>
#include<string>


namespace RAJA {

/*!
 * \brief Strongly typed "integer" class.
 *
 * Allows integers to be associated with a type, and disallows automatic
 * conversion.
 *
 * Useful for maintaining correctness in multidimensional loops and arrays.
 *
 * Use the RAJA_INDEX_VALUE(NAME) macro to define new indices.
 *
 * Yes, this uses the curiously-recurring template pattern.
 */
template<typename TYPE>
class IndexValue {

  public:
    /*!
     * \brief Default constructor initializes value to 0.
     */
    RAJA_HOST_DEVICE
    RAJA_INLINE
    IndexValue() : value(0) {}

    /*!
     * \brief Explicit constructor.
     * \param v   Initial value
     */
    RAJA_HOST_DEVICE
    RAJA_INLINE
    explicit IndexValue(Index_type v) : value(v) {}

    /*!
     * \brief Dereference provides cast-to-integer.
     */
    RAJA_HOST_DEVICE
    RAJA_INLINE
    Index_type operator*(void) const {return value;}



    RAJA_HOST_DEVICE
    RAJA_INLINE
    TYPE &operator++(int){
      value++;
      return *static_cast<TYPE *>(this);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE
    TYPE &operator++(){
      value++;
      return *static_cast<TYPE *>(this);
    }


    RAJA_HOST_DEVICE
    RAJA_INLINE
    TYPE &operator--(int){
      value--;
      return *static_cast<TYPE *>(this);
    }


    RAJA_HOST_DEVICE
    RAJA_INLINE
    TYPE &operator--(){
      value--;
      return *static_cast<TYPE *>(this);
    }


    RAJA_HOST_DEVICE
    RAJA_INLINE
    TYPE operator+(Index_type a) const { return TYPE(value+a); }

    RAJA_HOST_DEVICE
    RAJA_INLINE
    TYPE operator+(TYPE a) const { return TYPE(value+a.value); }

    RAJA_HOST_DEVICE
    RAJA_INLINE
    TYPE operator-(Index_type a) const { return TYPE(value-a); }

    RAJA_HOST_DEVICE
    RAJA_INLINE
    TYPE operator-(TYPE a) const { return TYPE(value-a.value); }

    RAJA_HOST_DEVICE
    RAJA_INLINE
    TYPE operator*(Index_type a) const { return TYPE(value*a); }

    RAJA_HOST_DEVICE
    RAJA_INLINE
    TYPE operator*(TYPE a) const { return TYPE(value*a.value); }

    RAJA_HOST_DEVICE
    RAJA_INLINE
    TYPE operator/(Index_type a) const { return TYPE(value/a); }

    RAJA_HOST_DEVICE
    RAJA_INLINE
    TYPE operator/(TYPE a) const { return TYPE(value/a.value); }


    RAJA_HOST_DEVICE
    RAJA_INLINE TYPE &operator+=(Index_type x){
      value += x;
      return *static_cast<TYPE *>(this);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE TYPE &operator+=(TYPE x){
      value += x.value;
      return *static_cast<TYPE *>(this);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE TYPE &operator-=(Index_type x){
      value -= x;
      return *static_cast<TYPE *>(this);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE TYPE &operator-=(TYPE x){
      value -= x.value;
      return *static_cast<TYPE *>(this);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE TYPE &operator*=(Index_type x){
      value *= x;
      return *static_cast<TYPE *>(this);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE TYPE &operator*=(TYPE x){
      value *= x.value;
      return *static_cast<TYPE *>(this);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE TYPE &operator/=(Index_type x){
      value /= x;
      return *static_cast<TYPE *>(this);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE TYPE &operator/=(TYPE x){
      value /= x.value;
      return *static_cast<TYPE *>(this);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE bool operator<(Index_type x) const {
      return( value < x);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE bool operator<(TYPE x) const {
      return( value < x.value);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE bool operator<=(Index_type x) const {
      return( value <= x);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE bool operator<=(TYPE x) const {
      return( value <= x.value);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE bool operator>(Index_type x) const {
      return( value > x);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE bool operator>(TYPE x) const {
      return( value > x.value);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE bool operator>=(Index_type x) const {
      return( value >= x);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE bool operator>=(TYPE x) const {
      return( value >= x.value);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE bool operator==(Index_type x) const {
      return( value == x);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE bool operator==(TYPE x) const {
      return( value == x.value);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE bool operator!=(Index_type x) const {
      return( value != x);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE bool operator!=(TYPE x) const {
      return( value != x.value);
    }


    // This is not implemented... but should be by the derived type
    // this is done by the macro
    static std::string getName(void);
  
  private:
    Index_type value;

};


/*!
 * \brief Helper class for convertIndex, since functions cannot be partially
 * specialized
 */
template<typename TO, typename FROM>
struct ConvertIndexHelper {
  RAJA_HOST_DEVICE
  RAJA_INLINE
  static TO convert(FROM val){
    return TO(*val);
  }
};

template<typename TO>
struct ConvertIndexHelper<TO, Index_type> {
  RAJA_HOST_DEVICE
  RAJA_INLINE
  static TO convert(Index_type val){
    return TO(val);
  }
};

/*!
 * \brief Function provides a way to take either an int or any Index<> type, and
 * convert it to another type, possibly another Index or an int.
 */
template<typename TO, typename FROM>
RAJA_HOST_DEVICE
RAJA_INLINE
TO convertIndex(FROM val){
  return ConvertIndexHelper<TO, FROM>::convert(val);
}




} // namespace RAJA



/*!
 * \brief Helper Macro to create new Index types.
 */
#define RAJA_INDEX_VALUE(TYPE, NAME) \
  class TYPE : public RAJA::IndexValue<TYPE>{ \
  public: \
    RAJA_HOST_DEVICE \
    RAJA_INLINE \
    explicit TYPE(RAJA::Index_type v) : RAJA::IndexValue<TYPE>::IndexValue(v) {} \
    static inline std::string getName(void){return NAME;} \
  };


#endif



