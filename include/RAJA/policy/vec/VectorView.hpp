/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining a multi-dimensional view class.
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

#ifndef RAJA_policy_vec_vectorview_HPP
#define RAJA_policy_vec_vectorview_HPP

#include <type_traits>

#include "RAJA/config.hpp"
#include "RAJA/util/View.hpp"

namespace RAJA
{

namespace internal {

template<typename VectorType>
struct MakeVectorHelper;

template<typename T, size_t N, size_t U>
struct MakeVectorHelper<RAJA::vec::Vector<T, N, U>&> {
    using vector_type = RAJA::vec::Vector<T, N, U>;

    RAJA_INLINE
    static
    vector_type &make_vector(T &ref, size_t ) {
      return reinterpret_cast<vector_type &>(ref);
    }
};

template<typename T, size_t N, size_t U>
struct MakeVectorHelper<RAJA::vec::StridedVector<T, N, U>> {
    using vector_type = RAJA::vec::StridedVector<T, N, U>;

    RAJA_INLINE
    static
    vector_type make_vector(T &ref, size_t stride) {
      return vector_type {&ref, stride};
    }
};

template<typename T>
struct StripVectorIndexHelper{

    using value_type = T;

    RAJA_INLINE
    static
    constexpr
    value_type strip(T const &v){
      return v;
    }
};

template<typename IdxType, typename VecType>
struct StripVectorIndexHelper<RAJA::vec::VectorIndex<IdxType, VecType>>{

    using vec_idx_t = RAJA::vec::VectorIndex<IdxType, VecType>;

    using value_type = IdxType;

    RAJA_INLINE
    static
    constexpr
    value_type strip(vec_idx_t const &idx) {
      return idx.value;
    }
};


template<typename T>
auto strip_vector_index(T const &idx) ->
  typename internal::StripVectorIndexHelper<T>::value_type
{
  return internal::StripVectorIndexHelper<T>::strip(idx);
}



template<typename T, RAJA::idx_t Stride1Dim, RAJA::idx_t I, typename ... Args>
struct GetVectorTypeHelper; 

// overload for non-vector index
template<typename T, RAJA::idx_t Stride1Dim, RAJA::idx_t I, typename Next, typename ... Rest>
struct GetVectorTypeHelper<T, Stride1Dim, I, Next, Rest...>{ 

  // next dimension helper
  using next_t = GetVectorTypeHelper<T, Stride1Dim, I+1, Rest...>;

  // Assign vector info to assume remaining in Rest...
  using type = typename next_t::type;
  static constexpr bool is_vector = next_t::is_vector;
  static constexpr RAJA::idx_t vector_dim = next_t::vector_dim;
  
};


// overload for vector index
template<typename T, RAJA::idx_t Stride1Dim, RAJA::idx_t I, typename IdxType, typename VecType, typename ... Rest>
struct GetVectorTypeHelper<T, Stride1Dim, I, RAJA::vec::VectorIndex<IdxType, VecType>, Rest...>{ 

  // next dimension helper
  using next_t = GetVectorTypeHelper<T, Stride1Dim, I+1, Rest...>;

  // is this dimension requesting a vector type?
  static constexpr bool is_this_vector = VecType::num_total_elements > 1; 

  // IF this dimension is a vector type, is it packed or strided?
  using vector_type = typename std::conditional<Stride1Dim==I, VecType &, typename VecType::strided_type>::type;

  // Assign either a vector type, or defer to the next dimension
  using type = typename std::conditional<is_this_vector, vector_type, typename next_t::type>::type;
  static constexpr bool is_vector = is_this_vector || next_t::is_vector;
  static constexpr RAJA::idx_t vector_dim = is_this_vector ? I : next_t::vector_dim;
  
  // check that we have at most one vector index
  static_assert(next_t::is_vector == false, "VectorView can only be indexed by a single vector index");
};

/**
* Termination case, assume its scalar
*/
template<typename T, RAJA::idx_t Stride1Dim, RAJA::idx_t I>
struct GetVectorTypeHelper<T, Stride1Dim, I>{
  using type = RAJA::vec::Vector<T, 1, 1> &;
  static constexpr bool is_vector = false;
  static constexpr RAJA::idx_t vector_dim = 0;
};


template<typename T, RAJA::idx_t Stride1Dim, typename ... Args>
using getVectorType = typename GetVectorTypeHelper<T, Stride1Dim, 0, Args...>::type;

template<typename T, RAJA::idx_t Stride1Dim, typename ... Args>
RAJA_INLINE
constexpr
RAJA::idx_t getVectorDim(){
  return GetVectorTypeHelper<T, Stride1Dim, 0, Args...>::vector_dim;
}


} // namespace internal

template <typename ViewType>
struct VectorViewWrapper{
  using base_type = ViewType;
  using pointer_type = typename base_type::pointer_type;
  using value_type = typename base_type::value_type;
  using layout_type = typename base_type::layout_type;

  // Determine the stride1 dimension of this views layout so we know when to return
  // a packed vs strided vector object
  static constexpr RAJA::idx_t stride1_dim = ViewType::layout_type::stride1_dim;


  base_type base_;

  RAJA_INLINE
  constexpr explicit VectorViewWrapper(ViewType const &view) : base_{view} {}

  RAJA_INLINE layout_type const &get_layout() const { return base_.get_layout(); }

  RAJA_INLINE void set_data(pointer_type data_ptr) { base_.set_data(data_ptr); }

  template <typename... ARGS>
  RAJA_HOST_DEVICE RAJA_INLINE auto operator()(ARGS &&... args) const ->
    internal::getVectorType<value_type, stride1_dim, camp::decay<ARGS>...>
  {   
    // Figure out the type of vector object to be returned: scalar vector, packed vector, or strided vector
    using vector_type = internal::getVectorType<value_type, stride1_dim, camp::decay<ARGS>...>;

    // determine which dimension is getting packed into a vector
    static constexpr RAJA::idx_t vector_dim = internal::getVectorDim<value_type, stride1_dim, camp::decay<ARGS>...>();

    // get the actual value/pointer from the underlying view
    auto &val = base_.operator()(internal::strip_vector_index(args)...);

    // Create a vector reference or object depeding on the vector_type
    return internal::MakeVectorHelper<vector_type>::make_vector(val, get_layout().strides[vector_dim]);
  }


};


/**
  Convenience function that produces a VectorView wrapper around an existing view.
*/
template <typename ViewType>
RAJA_INLINE VectorViewWrapper<ViewType> make_vector_view(
    ViewType const &view)
{
  return RAJA::VectorViewWrapper<ViewType>(view);
}

/**
  Convenience function that produces a VectorView from a raw pointer.
*/
template<typename T>
RAJA_INLINE
VectorViewWrapper<RAJA::View<T, RAJA::Layout<1, RAJA::Index_type, 0> > > make_vector_view(T *data)
{
  using ViewType = RAJA::View<T, RAJA::Layout<1, RAJA::Index_type, 0> >;
  ViewType view(data, 1);
  return RAJA::VectorViewWrapper<ViewType>(view);
}


}  // namespace RAJA

#endif
