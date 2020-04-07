/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining SIMD/SIMT register operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_vector_tensorindex_HPP
#define RAJA_pattern_vector_tensorindex_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"


namespace RAJA
{

  namespace internal {
    template<typename ARG>
    struct TensorIndexTraits;
  }

  template<typename IDX, typename TENSOR_TYPE, camp::idx_t DIM>
  class TensorIndex {
    public:
      using self_type = TensorIndex<IDX, TENSOR_TYPE, DIM>;
      using value_type = strip_index_type_t<IDX>;
      using index_type = IDX;
      using tensor_type = TENSOR_TYPE;
      using tensor_traits = internal::TensorIndexTraits<self_type>;

      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      TensorIndex() : m_index(index_type(0)), m_length(tensor_traits::num_elem()) {}


      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      explicit TensorIndex(index_type value) : m_index(value), m_length(tensor_traits::num_elem()) {}


      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      TensorIndex(index_type value, camp::idx_t length) : m_index(value), m_length(length) {}

      template<typename T, camp::idx_t D>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      TensorIndex(TensorIndex<IDX, T, D> const &c) : m_index(*c), m_length(c.size()) {}


      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      index_type const &operator*() const {
        return m_index;
      }

      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      camp::idx_t size() const {
        return m_length;
      }

      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      camp::idx_t dim() const {
        return DIM;
      }

    private:
      index_type m_index;
      camp::idx_t m_length;
  };


  /*!
   * Index that specifies the starting element index of a Vector
   */
  template<typename IDX, typename VECTOR_TYPE>
  using VectorIndex =  TensorIndex<IDX, VECTOR_TYPE, 0>;

  /*!
   * Index that specifies the starting Row index of a matrix
   */
  template<typename IDX, typename MATRIX_TYPE>
  using RowIndex =  TensorIndex<IDX, MATRIX_TYPE, 0>;

  /*!
   * Index that specifies the starting Column index of a matrix
   */
  template<typename IDX, typename MATRIX_TYPE>
  using ColIndex =  TensorIndex<IDX, MATRIX_TYPE, 1>;




  namespace internal{


    /* Partial specialization for the strip_index_type_t helper in
       IndexValue.hpp
    */
    template<typename IDX, typename VECTOR_TYPE, camp::idx_t DIM>
    struct StripIndexTypeT<TensorIndex<IDX, VECTOR_TYPE, DIM>>
    {
        using type = typename TensorIndex<IDX, VECTOR_TYPE, DIM>::value_type;
    };


    // Helper that strips the Vector type from an argument
    template<typename ARG>
    struct TensorIndexTraits {
        using arg_type = ARG;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        bool isTensorIndex(){
          return false;
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        arg_type const &strip(arg_type const &arg){
          return arg;
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        camp::idx_t size(arg_type const &){
          return 1;
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        camp::idx_t dim(){
          return 0;
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        camp::idx_t num_elem(){
          return 1;
        }
    };

    template<typename IDX, typename TENSOR_TYPE, camp::idx_t DIM>
    struct TensorIndexTraits<TensorIndex<IDX, TENSOR_TYPE, DIM>> {
        using index_type = TensorIndex<IDX, TENSOR_TYPE, DIM>;
        using arg_type = IDX;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        bool isTensorIndex(){
          return true;
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        arg_type const &strip(index_type const &arg){
          return *arg;
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        camp::idx_t size(index_type const &arg){
          return arg.size();
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        camp::idx_t dim(){
          return DIM;
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        camp::idx_t num_elem(){
          return TENSOR_TYPE::s_dim_elem(DIM);
        }
    };

    /*
     * Returns vector size of argument.
     *
     * For scalars, always returns 1.
     *
     * For VectorIndex types, returns the number of vector lanes.
     */
    template<typename ARG>
    RAJA_INLINE
    RAJA_HOST_DEVICE
    constexpr
    bool isTensorIndex()
    {
      return TensorIndexTraits<ARG>::isTensorIndex();
    }

    template<typename ARG>
    RAJA_INLINE
    RAJA_HOST_DEVICE
    constexpr
    auto stripTensorIndex(ARG const &arg) ->
    typename TensorIndexTraits<ARG>::arg_type const &
    {
      return TensorIndexTraits<ARG>::strip(arg);
    }

    /*
     * Returns vector size of argument.
     *
     * For scalars, always returns 1.
     *
     * For VectorIndex types, returns the number of vector lanes.
     */
    template<typename ARG>
    RAJA_INLINE
    RAJA_HOST_DEVICE
    constexpr
    camp::idx_t getTensorSize(ARG const &arg)
    {
      return TensorIndexTraits<ARG>::size(arg);
    }

    /*
     * Returns vector dim of argument.
     *
     * For scalars, always returns 0.
     *
     * For VectorIndex types, returns the DIM argument.
     * For vector_exec, this is always 0
     *
     * For matrices, DIM means:
     *   0 : Row
     *   1 : Column
     */
    template<typename ARG>
    RAJA_INLINE
    RAJA_HOST_DEVICE
    constexpr
    camp::idx_t getTensorDim()
    {
      return TensorIndexTraits<ARG>::dim();
    }

    /*
     * Lambda<N, Seg<X>>  overload that matches VectorIndex types, and properly
     * includes the vector length with them
     */
    template<typename IDX, typename TENSOR_TYPE, camp::idx_t DIM, camp::idx_t id>
    struct LambdaSegExtractor<TensorIndex<IDX, TENSOR_TYPE, DIM>, id>
    {

      template<typename Data>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      static TensorIndex<IDX, TENSOR_TYPE, DIM> extract(Data &&data)
      {
        return TensorIndex<IDX, TENSOR_TYPE, DIM>(
            camp::get<id>(data.segment_tuple).begin()[camp::get<id>(data.offset_tuple)],
            camp::get<id>(data.vector_sizes));
      }

    };

    /*
     * Lambda<N, Seg<X>>  overload that matches VectorIndex types, and properly
     * includes the vector length with them
     */
    template<typename IDX, typename TENSOR_TYPE, camp::idx_t DIM, camp::idx_t id>
    struct LambdaOffsetExtractor<TensorIndex<IDX, TENSOR_TYPE, DIM>, id>
    {

      template<typename Data>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      static TensorIndex<IDX, TENSOR_TYPE, DIM> extract(Data &&data)
      {
        return TensorIndex<IDX, TENSOR_TYPE, DIM>(
            IDX(camp::get<id>(data.offset_tuple)), // convert offset type to IDX
            camp::get<id>(data.vector_sizes));
      }

    };

  } // namespace internal



}  // namespace RAJA


#endif
