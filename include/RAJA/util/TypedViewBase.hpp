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
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_util_TypedViewBase_HPP
#define RAJA_util_TypedViewBase_HPP

#include <type_traits>

#include "RAJA/config.hpp"

#include "RAJA/pattern/atomic.hpp"

#include "RAJA/util/Layout.hpp"
#include "RAJA/util/OffsetLayout.hpp"

namespace RAJA
{

namespace internal
{

  template<camp::idx_t, typename T>
  struct IndexToType{
      using type = T;
  };

  template<typename IdxSeq, typename T>
  struct SequenceToType;

  template<camp::idx_t ... Perm, typename T>
  struct SequenceToType<camp::idx_seq<Perm...>, T>{
      using type =  camp::list<typename IndexToType<Perm, T>::type...>;
  };

  template<typename Perm>
  using getDefaultIndexTypes = typename SequenceToType<Perm, RAJA::Index_type>::type;




  //Helpers to convert
  //layouts -> OffsetLayouts
  //Typedlayouts -> TypedOffsetLayouts
  template<typename layout>
  struct add_offset
  {
    using type = RAJA::OffsetLayout<layout::n_dims>;
  };

  template<typename IdxLin, typename...DimTypes>
  struct add_offset<RAJA::TypedLayout<IdxLin,camp::tuple<DimTypes...>>>
  {
    using type = RAJA::TypedOffsetLayout<IdxLin,camp::tuple<DimTypes...>>;
  };




  /*
   * Returns the number of arguments which are VectorIndexs
   */
  template<typename ... ARGS>
  RAJA_INLINE
  RAJA_HOST_DEVICE
  static constexpr camp::idx_t count_num_tensor_args(){
		return 0;
  }



  namespace detail {

  /*!
   * Provides conversion of view data to a return type.
   *
   * For scalars, this just returns the scalar.
   *
   * In the future development, this may return SIMD vectors or matrices using
   * class specializations.
   */
  template<camp::idx_t NumVectors, typename Args, typename ElementType, typename PointerType, typename LinIdx, camp::idx_t StrideOneDim>
  struct ViewReturnHelper
  {
      static_assert(NumVectors == 0, "Vectors and Matrices not supported yet");
  };


  /*
   * Specialization for Scalar return types
   */
  template<typename ... Args, typename ElementType, typename PointerType, typename LinIdx, camp::idx_t StrideOneDim>
  struct ViewReturnHelper<0, camp::list<Args...>, ElementType, PointerType, LinIdx, StrideOneDim>
  {
      using return_type = ElementType &;

      template<typename LayoutType>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      constexpr
      return_type make_return(LayoutType const &layout, PointerType const &data, Args const &... args){
        return data[stripIndexType(layout(args...))];
      }
  };



  } // namespace detail


  /*
   * Computes the return type of a view.
   *
   * If any of the arguments are a VectorIndex, it creates a VectorRef
   * return type.
   *
   * Otherwise it produces the usual scalar reference return type
   */
  template<typename ElementType, typename PointerType, typename LinIdx, typename LayoutType, typename ... Args>
  using view_return_type_t =
      typename detail::ViewReturnHelper<
        count_num_tensor_args<Args...>(),
        camp::list<Args...>,
        ElementType,
        PointerType,
        LinIdx,
        LayoutType::stride_one_dim>::return_type;

  /*
   * Creates the return value for a View
   *
   * If any of the arguments are a VectorIndex, it creates a VectorRef
   * return value.
   *
   * Otherwise it produces the usual scalar reference return value
   */
  template<typename ElementType, typename LinIdx, typename LayoutType, typename PointerType, typename ... Args>
  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr
  view_return_type_t<ElementType, PointerType, LinIdx, LayoutType, Args...>
  view_make_return_value(LayoutType const &layout, PointerType const &data, Args const &... args){
    return detail::ViewReturnHelper<
        count_num_tensor_args<Args...>(),
        camp::list<Args...>,
        ElementType,
        PointerType,
        LinIdx,
        LayoutType::stride_one_dim>::make_return(layout, data, args...);
  }



  namespace detail
  {

  /**
   * This class will help strip strongly typed indices
   *
   * This default implementation static_asserts that Expected==Arg, otherwise
   * it's an error.  This enforces types for the TypedView.
   *
   * Specialization where expected type is same as argument type.
   * In this case, there is no VectorIndex to unpack, just strip any strongly
   * typed indices.
   */
  template<typename Expected, typename Arg>
  struct MatchTypedViewArgHelper{
    static_assert(std::is_convertible<Arg, Expected>::value,
        "Argument isn't compatible");

    using type = strip_index_type_t<Arg>;

    static RAJA_HOST_DEVICE RAJA_INLINE
    constexpr
    type extract(Arg arg){
      return stripIndexType(arg);
    }
  };


  } //namespace detail


  template<typename Expected, typename Arg>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  constexpr
  typename detail::MatchTypedViewArgHelper<Expected, Arg>::type
  match_typed_view_arg(Arg const &arg)
  {
    return detail::MatchTypedViewArgHelper<Expected, Arg>::extract(arg);
  }



template <typename ValueType,
          typename PointerType,
          typename LayoutType>
class ViewBase {

  public:
    using value_type = ValueType;
    using pointer_type = PointerType;
    using layout_type = LayoutType;
    using linear_index_type = typename layout_type::IndexLinear;
    using nc_value_type = typename std::remove_const<value_type>::type;
    using nc_pointer_type = typename std::add_pointer<typename std::remove_const<
        typename std::remove_pointer<pointer_type>::type>::type>::type;

    using Self = ViewBase<value_type, pointer_type, layout_type>;
    using NonConstView = ViewBase<nc_value_type, nc_pointer_type, layout_type>;

    using shifted_layout_type = typename add_offset<layout_type>::type;
    using ShiftedView = ViewBase<value_type, pointer_type, shifted_layout_type>;

  protected:
    pointer_type m_data;
    layout_type const m_layout;

  public:


    /*
     * Defaulted operators (AJK):
     *
     * OpenMP Target currently needs the View classes to be trivially copyable,
     * which means that we need to use the default ctor's and assignment
     * operators.
     *
     * These defaulted operators cause issues with some versions of CUDA, so
     * in the case that CUDA is enabled, we switch to explicitly defined
     * operators.
     */
#if (defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_CLANG_CUDA))
    RAJA_HOST_DEVICE
    RAJA_INLINE
    constexpr ViewBase(){};

    RAJA_HOST_DEVICE
    RAJA_INLINE ViewBase(ViewBase const &c)
      : m_layout(c.m_layout), m_data(c.m_data)
    {
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE
    ViewBase &operator=(ViewBase const &c)
    {
      m_layout = c.m_layout;
      m_data = c.m_data;
    }
#else
    constexpr ViewBase() = default;
    RAJA_INLINE constexpr ViewBase(ViewBase const &) = default;
    RAJA_INLINE constexpr ViewBase(ViewBase &&) = default;
    RAJA_INLINE ViewBase& operator=(ViewBase const &) = default;
    RAJA_INLINE ViewBase& operator=(ViewBase &&) = default;

#endif

    RAJA_HOST_DEVICE
    RAJA_INLINE
    constexpr
    ViewBase(pointer_type data, layout_type &&layout) :
    m_data(data), m_layout(layout)
    {
    }

    template <typename... Args>
    RAJA_HOST_DEVICE
    RAJA_INLINE
    constexpr
    ViewBase(pointer_type data, Args... dim_sizes) :
    m_data(data), m_layout(dim_sizes...)
    {
    }


    template <bool IsConstView = std::is_const<value_type>::value>
    RAJA_HOST_DEVICE
    RAJA_INLINE
    constexpr
    ViewBase(typename std::enable_if<IsConstView, NonConstView>::type const &rhs) :
    m_data(rhs.get_data()), m_layout(rhs.get_layout())
    {
    }


    RAJA_HOST_DEVICE
    RAJA_INLINE void set_data(PointerType data_ptr){
      m_data = data_ptr;
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE
    constexpr
    pointer_type const &get_data() const
    {
      return m_data;
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE
    constexpr
    layout_type const &get_layout() const
    {
      return m_layout;
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE
    constexpr
    camp::idx_t size() const
    {
      return m_layout.size();
    }


    template <typename... Args>
    RAJA_HOST_DEVICE
    RAJA_INLINE
    constexpr
    view_return_type_t<value_type, pointer_type, linear_index_type, layout_type, Args...>
    operator()(Args... args) const
    {
      return view_make_return_value<value_type, linear_index_type>(m_layout, m_data, args...);
    }



    /*
     * Compatibility note (AJK):
     * We are using variadic arguments even though operator[] takes exactly 1 argument
     * This gets around a template instantiation bug in CUDA/nvcc 9.1, which seems to have
     * been fixed in CUDA 9.2+
     */
    template <typename ... Args>
    RAJA_HOST_DEVICE
    RAJA_INLINE
    constexpr
    view_return_type_t<value_type, pointer_type, linear_index_type, layout_type, Args...>
    operator[](Args ... args) const
    {
      return view_make_return_value<value_type, linear_index_type>(m_layout, m_data, args...);
    }



    template <size_t n_dims = layout_type::n_dims, typename IdxLin = linear_index_type>
    RAJA_INLINE
    ShiftedView shift(const std::array<IdxLin, n_dims>& shift)
    {
      static_assert(n_dims==layout_type::n_dims, "Dimension mismatch in view shift");

      shifted_layout_type shift_layout(m_layout);
      shift_layout.shift(shift);

      return ShiftedView(m_data, shift_layout);
    }

};


template <typename ValueType,
        typename PointerType,
        typename LayoutType,
        typename IndexTypes>
class TypedViewBase;

template <typename ValueType,
          typename PointerType,
          typename LayoutType,
          typename... IndexTypes>
class TypedViewBase<ValueType, PointerType, LayoutType, camp::list<IndexTypes...>> :
  public ViewBase<ValueType, PointerType, LayoutType>
{

  public:
    using value_type = ValueType;
    using pointer_type = PointerType;
    using layout_type = LayoutType;
    using linear_index_type = typename layout_type::IndexLinear;
    using nc_value_type = typename std::remove_const<value_type>::type;
    using nc_pointer_type = typename std::add_pointer<typename std::remove_const<
        typename std::remove_pointer<pointer_type>::type>::type>::type;

    using Base = ViewBase<ValueType, PointerType, LayoutType>;
    using Self = TypedViewBase<value_type, pointer_type, layout_type, camp::list<IndexTypes...> >;
    using NonConstView = TypedViewBase<nc_value_type, nc_pointer_type, layout_type, camp::list<IndexTypes...> >;

    using shifted_layout_type = typename add_offset<layout_type>::type;
    using ShiftedView = TypedViewBase<value_type, pointer_type, shifted_layout_type, camp::list<IndexTypes...> >;

    static constexpr size_t n_dims = sizeof...(IndexTypes);

    using Base::Base;

    template <typename... Args>
    RAJA_HOST_DEVICE
    RAJA_INLINE
    constexpr
    view_return_type_t<value_type, pointer_type, linear_index_type, layout_type, Args...>
    operator()(Args... args) const
    {
      return view_make_return_value<value_type, linear_index_type>(Base::m_layout, Base::m_data, match_typed_view_arg<IndexTypes>(args)...);
    }



    /*
     * Compatibility note (AJK):
     * We are using variadic arguments even though operator[] takes exactly 1 argument
     * This gets around a template instantiation bug in CUDA/nvcc 9.1, which seems to have
     * been fixed in CUDA 9.2+
     */
    template <typename ... Args>
    RAJA_HOST_DEVICE
    RAJA_INLINE
    constexpr
    view_return_type_t<value_type, pointer_type, linear_index_type, layout_type, Args...>
    operator[](Args ... args) const
    {
      return view_make_return_value<value_type, linear_index_type>(Base::m_layout, Base::m_data, match_typed_view_arg<IndexTypes>(args)...);
    }



    template <size_t n_dims = sizeof...(IndexTypes), typename IdxLin = linear_index_type>
    RAJA_INLINE
    ShiftedView shift(const std::array<IdxLin, n_dims>& shift)
    {
      static_assert(n_dims==layout_type::n_dims, "Dimension mismatch in view shift");

      shifted_layout_type shift_layout(Base::get_layout());
      shift_layout.shift(shift);

      return ShiftedView(Base::get_data(), shift_layout);
    }

};



} // namespace internal

}  // namespace RAJA

#endif
