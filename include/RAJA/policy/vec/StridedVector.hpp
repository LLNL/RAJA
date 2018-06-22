/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing a SIMD vector gather/scatter abstraction
 *          that allows reading and writing of strided data to/from vector 
 *          registers
 *
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_vec_stridedvector_HPP
#define RAJA_policy_vec_stridedvector_HPP

#include <iterator>
#include <type_traits>

#include "RAJA/config.hpp"

#include "RAJA/util/types.hpp"

#include "RAJA/policy/vec/Vector.hpp"

namespace RAJA
{

namespace vec
{



template<typename T, size_t N, size_t U>
struct StridedVector
{
  using self_t = StridedVector<T, N, U>;

  static constexpr size_t element_width = N*U;

  using rel_vec_t = Vector<T, N, U>;

  T *data;
  size_t stride;

  RAJA_INLINE
  operator rel_vec_t () const {
    return to_vector();
  }

  RAJA_INLINE
  rel_vec_t to_vector() const {
    rel_vec_t x;

    for(size_t u = 0;u < U;++ u){
      for(size_t n = 0;n < N; ++ n){
        x.value[u][n] = data[(u*N + n)*stride];
      }
    }

    return x;
  }

  RAJA_INLINE
  self_t &operator=(rel_vec_t const &a){

    for(size_t u = 0;u < U;++ u){
      for(size_t n = 0;n < N; ++ n){
        data[(u*N + n)*stride] = a.value[u][n];
      }
    }

    return *this;
  }

  RAJA_INLINE
  self_t &operator=(T const &a){

    data[0] = a;

    return *this;
  }


  RAJA_INLINE
  self_t &operator+=(rel_vec_t const &a){

    for(size_t u = 0;u < U;++ u){
      for(size_t n = 0;n < N; ++ n){
        data[(u*N + n)*stride] += a.value[u][n];
      }
    }

    return *this;
  }


  RAJA_INLINE
  self_t &operator-=(rel_vec_t const &a){

    for(size_t u = 0;u < U;++ u){
      for(size_t n = 0;n < N; ++ n){
        data[(u*N + n)*stride] -= a.value[u][n];
      }
    }

    return *this;
  }

  RAJA_INLINE
  self_t &operator*=(rel_vec_t const &a){

    for(size_t u = 0;u < U;++ u){
      for(size_t n = 0;n < N; ++ n){
        data[(u*N + n)*stride] *= a.value[u][n];
      }
    }

    return *this;
  }


  RAJA_INLINE
  self_t &operator/=(rel_vec_t const &a){

    for(size_t u = 0;u < U;++ u){
      for(size_t n = 0;n < N; ++ n){
        data[(u*N + n)*stride] /= a.value[u][n];
      }
    }

    return *this;
  }



  RAJA_INLINE
  rel_vec_t operator+(rel_vec_t const &a) const {
    return to_vector()+a;
  }

  RAJA_INLINE
  rel_vec_t operator-(rel_vec_t const &a) const {
    return to_vector()-a;
  }

  RAJA_INLINE
  rel_vec_t operator*(rel_vec_t const &a) const {
    return to_vector()*a;
  }

  RAJA_INLINE
  rel_vec_t operator/(rel_vec_t const &a) const {
    return to_vector()/a;
  }



  RAJA_INLINE
  T sum() const {
    return to_vector().sum();
  }

  RAJA_INLINE
  T product() const {
    return to_vector().product();
  }

  RAJA_INLINE
  T min() const {
    return to_vector().min();
  }

  RAJA_INLINE
  T max() const {
    return to_vector().max();
  }

};



}  // closing brace for vec namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
