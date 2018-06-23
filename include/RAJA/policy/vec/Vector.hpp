/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for a portable SIMD vector register abstraction
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

#ifndef RAJA_policy_vec_vector_HPP
#define RAJA_policy_vec_vector_HPP

#include <iterator>
#include <type_traits>

#include "RAJA/config.hpp"

#include "RAJA/util/types.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/policy/vec/VectorRegister.hpp"

namespace RAJA
{

namespace vec
{

// forward declare (see StridedVector.hpp)
template<typename T, size_t N, size_t U>
struct StridedVector;

template<typename T, size_t N, size_t U>
struct Vector
{
    using self_t = Vector<T, N, U>;
    using strided_type = StridedVector<T,N,U>;
    using scalar_type = T;

    static constexpr size_t num_vec_elements = N;
    static constexpr size_t num_vec_bytes = N*sizeof(T);
    static constexpr size_t num_unroll = U;

    static constexpr size_t num_total_elements = N*U;
    static constexpr size_t num_total_bytes = N*U*sizeof(T);


    // Make sure we use a non-vector type for a scalar in the case we have
    // a vector width of 1
    using value_type =
        typename std::conditional<N==1, T, internal::VectorRegister<T,N>>::type;

    value_type value[U];


    RAJA_INLINE
    Vector(){}

    RAJA_INLINE
    Vector(T val){
      for(size_t i = 0;i < U;++ i){
        value[i] = val;
      }
    }

    RAJA_INLINE
    Vector(Vector<T,1,1> val){
      for(size_t i = 0;i < U;++ i){
        value[i] = 0;
      }
      value[0][0] = val.value[0];
    }


    RAJA_INLINE
    self_t &operator=(self_t const &a){

      for(size_t i = 0;i < U;++ i){
        value[i] = a.value[i];
      }

      return *this;
    }


    RAJA_INLINE
    self_t &operator=(T a){

      for(size_t i = 0;i < U;++ i){
        value[i] = a;
      }

      return *this;
    }

    RAJA_INLINE
    bool operator!=(self_t const &a) const{
      bool equals = false;

      for(size_t u = 0;u < U;++ u){
        for(size_t n = 0;n < N;++ n){
          equals = equals && (value[u][n] == a.value[u][n]);
        }
      }

      return !equals;
    }

    RAJA_INLINE
    self_t operator+(self_t const &a) const{
      self_t result;

      for(size_t i = 0;i < U;++ i){
        result.value[i] = value[i] + a.value[i];
      }

      return result;
    }

    RAJA_INLINE
    self_t operator-(self_t const &a) const{
      self_t result;

      for(size_t i = 0;i < U;++ i){
        result.value[i] = value[i] - a.value[i];
      }

      return result;
    }

    RAJA_INLINE
    self_t operator*(self_t const &a) const{
      self_t result;

      for(size_t i = 0;i < U;++ i){
        result.value[i] = value[i] * a.value[i];
      }

      return result;
    }

    RAJA_INLINE
    self_t operator/(self_t const &a) const{
      self_t result;

      for(size_t i = 0;i < U;++ i){
        result.value[i] = value[i] / a.value[i];
      }

      return result;
    }

    RAJA_INLINE
    self_t &operator+=(self_t const &a){

      for(size_t i = 0;i < U;++ i){
        value[i] += a.value[i];
      }

      return *this;
    }

    RAJA_INLINE
    self_t &operator-=(self_t const &a){

      for(size_t i = 0;i < U;++ i){
        value[i] -= a.value[i];
      }

      return *this;
    }

    RAJA_INLINE
    self_t &operator*=(self_t const &a){

      for(size_t i = 0;i < U;++ i){
        value[i] *= a.value[i];
      }

      return *this;
    }

    RAJA_INLINE
    self_t &operator/=(self_t const &a){

      for(size_t i = 0;i < U;++ i){
        value[i] /= a.value[i];
      }

      return *this;
    }

    RAJA_INLINE
    T sum() const {

      T x = 0;

      for(size_t u = 0;u < U;++ u){
        for(size_t n = 0;n < N;++ n){
          x += value[u][n];
        }
      }

      return x;
    }

    RAJA_INLINE
    T product() const {

      T x = 1;

      for(size_t u = 0;u < U;++ u){
        for(size_t n = 0;n < N;++ n){
          x *= value[u][n];
        }
      }

      return x;
    }

    RAJA_INLINE
    T min() const {

      T x = value[0][0];

      for(size_t u = 0;u < U;++ u){
        for(size_t n = 0;n < N;++ n){
          x = value[u][n] < x ? value[u][n] : x;
        }
      }

      return x;
    }

    RAJA_INLINE
    T max() const {

      T x = value[0][0];

      for(size_t u = 0;u < U;++ u){
        for(size_t n = 0;n < N;++ n){
          x = value[u][n] > x ? value[u][n] : x;
        }
      }

      return x;
    }

};



}  // closing brace for vec namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
