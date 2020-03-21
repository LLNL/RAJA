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
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_util_TypedViewBase_HPP
#define RAJA_util_TypedViewBase_HPP

#include <type_traits>

#include "RAJA/config.hpp"

#include "RAJA/pattern/atomic.hpp"
#include "RAJA/pattern/vector.hpp"

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

  // Helper that strips the Vector type from an argument
  template<typename ARG>
  struct StripVectorIndex {
      using arg_type = ARG;
      using vector_type = RAJA::vector_scalar_register;
      static constexpr bool s_is_vector = false;


      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      constexpr
      arg_type const &get(arg_type const &arg){
        return arg;
      }
  };

  template<typename IDX, typename VECTOR_TYPE>
  struct StripVectorIndex<VectorIndex<IDX, VECTOR_TYPE>> {
      using arg_type = IDX;
      using vector_type = VECTOR_TYPE;
      static constexpr bool s_is_vector = true;

      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      constexpr
      arg_type const &get(VectorIndex<IDX, VECTOR_TYPE> const &arg){
        return *arg;
      }
  };

  template<typename ARG>
  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr
  auto stripVectorIndex(ARG const &arg) ->
  typename StripVectorIndex<ARG>::arg_type const &
  {
    return StripVectorIndex<ARG>::get(arg);
  }

  template<camp::idx_t I, typename ... ARGS>
  struct ExtractVectorArg;

  template<camp::idx_t I, typename ARG0, typename ... ARG_REST>
  struct ExtractVectorArg<I, ARG0, ARG_REST...>{
      using strip_index_t = StripVectorIndex<ARG0>;
      using next_t = ExtractVectorArg<I+1, ARG_REST...>;

      static constexpr camp::idx_t s_num_vector_args =
          (strip_index_t::s_is_vector ? 1 : 0) + next_t::s_num_vector_args;

      static constexpr camp::idx_t s_vector_arg_idx =
          (strip_index_t::s_is_vector ? I : next_t::s_vector_arg_idx);

      using vector_type =
          typename std::conditional<strip_index_t::s_is_vector,
          typename strip_index_t::vector_type,
          typename next_t::vector_type>::type;
  };

  // Termination case
  template<camp::idx_t I>
  struct ExtractVectorArg<I>{
      static constexpr camp::idx_t s_num_vector_args = 0;
      static constexpr camp::idx_t s_vector_arg_idx = -1;
      using vector_type = RAJA::vector_scalar_register;
  };

  // Helper to unpack VectorIndex
  template<typename IdxLin, typename ValueType, typename PointerType, typename ExtractType, bool IsVector>
  struct ViewVectorArgsHelper;

  template<typename IdxLin, typename ValueType, typename PointerType, typename ExtractType>
  struct ViewVectorArgsHelper<IdxLin, ValueType, PointerType, ExtractType, true> {

      // Count how many VectorIndex arguments there are
      static constexpr size_t s_num_vector_args = ExtractType::s_num_vector_args;

      // Make sure we don't have conflicting arguments
      static_assert(s_num_vector_args < 2, "View only supports a single VectorIndex at a time");


      // We cannot compute this yet.
      // TODO: figure out how this might be computed...
      static constexpr bool s_is_stride_one = false;


      // Compute a Vector type
      using vector_type = typename ExtractType::vector_type;

      using IndexLinear = strip_index_type_t<IdxLin>;

      using type = VectorRef<vector_type, IdxLin, PointerType, s_is_stride_one>;

      template<typename Args>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      constexpr
      type createReturn(IndexLinear lin_index, Args args, PointerType pointer, IndexLinear stride){
        return type(lin_index, camp::get<ExtractType::s_vector_arg_idx>(args).size(), pointer, stride);
      }
  };

  template<typename IdxLin, typename ValueType, typename PointerType, typename ExtractType>
  struct ViewVectorArgsHelper<IdxLin, ValueType, PointerType, ExtractType, false> {

      // We cannot compute this yet.
      // TODO: figure out how this might be computed...
      static constexpr bool s_is_stride_one = false;


      using IndexLinear = strip_index_type_t<IdxLin>;

      using type = ValueType&;

      template<typename Args>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      constexpr
      type createReturn(IndexLinear lin_index, Args , PointerType pointer, IndexLinear ){
        return pointer[lin_index];
      }
  };



  template<typename IdxLin, typename ValueType, typename PointerType, typename ... ARGS>
  using ViewVectorHelper = ViewVectorArgsHelper<IdxLin, ValueType, PointerType,
      ExtractVectorArg<0, ARGS...>, ExtractVectorArg<0, ARGS...>::s_num_vector_args >= 1>;



  /**
   * This class will help strip strongly typed
   *
   * This default implementation static_asserts that Expected==Arg, otherwise
   * it's an error.  This enforces types for the TypedView.
   *
   * Specialization where expected type is same as argument type.
   * In this case, there is no VectorIndex to unpack, just strip any strongly
   * typed indices.
   */
  template<typename Expected, typename Arg>
  struct TypedViewVectorHelper{
    static_assert(std::is_convertible<Arg, Expected>::value,
        "Argument isn't compatible");

    using type = strip_index_type_t<Arg>;

    static RAJA_HOST_DEVICE RAJA_INLINE
    type extract(Arg arg){
      return stripIndexType(arg);
    }
  };


  /**
   * Specialization where expected type is wrapped in a VectorIndex type
   *
   * In this case, there is no VectorIndex to unpack, just strip any strongly
   * typed indices.
   */
  template<typename Expected, typename Arg, typename VectorType>
  struct TypedViewVectorHelper<Expected, RAJA::VectorIndex<Arg, VectorType> >{

    static_assert(std::is_convertible<Arg, Expected>::value,
        "Argument isn't compatible");

    using arg_type = strip_index_type_t<Arg>;

    using type = RAJA::VectorIndex<arg_type, VectorType>;

    static constexpr RAJA_HOST_DEVICE RAJA_INLINE
    type extract(RAJA::VectorIndex<Arg, VectorType> vec_arg){
      return type(stripIndexType(*vec_arg), vec_arg.size());
    }
  };



  template<typename Expected, typename Arg>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  constexpr
  auto vectorArgExtractor(Arg const &arg) ->
  typename TypedViewVectorHelper<Expected, Arg>::type
  {
    return TypedViewVectorHelper<Expected, Arg>::extract(arg);
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

    static constexpr size_t n_dims = layout_type::n_dims;

  private:
    pointer_type m_data;
    layout_type const m_layout;

  public:
    RAJA_HOST_DEVICE
    constexpr
    RAJA_INLINE ViewBase() {}

    RAJA_HOST_DEVICE
    RAJA_INLINE
    explicit
    constexpr
    ViewBase(pointer_type data) : m_data(data)
    {
    }

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


  protected:

    // making this specifically typed would require unpacking the layout,
    // this is easier to maintain
    template <typename... Args>
    RAJA_HOST_DEVICE RAJA_INLINE
    constexpr
    auto operator_internal(Args... args) const ->
    typename internal::ViewVectorHelper<linear_index_type, value_type, pointer_type, Args...>::type
    {
      using helper_t = internal::ViewVectorHelper<linear_index_type, value_type, pointer_type, Args...>;

      return helper_t::createReturn(stripIndexType(m_layout(internal::stripVectorIndex(args)...)), camp::make_tuple(args...), m_data, 1);
    }

  public:

    template <typename... Args>
    RAJA_HOST_DEVICE
    RAJA_INLINE

    auto operator()(Args... args) const ->
    typename internal::ViewVectorHelper<linear_index_type, value_type, pointer_type, Args...>::type
    {
      return operator_internal(args...);
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
    auto operator[](Args ... args) const ->
    typename internal::ViewVectorHelper<linear_index_type, value_type, pointer_type, Args...>::type
    {
      return operator_internal(args...);
    }



    template <size_t N = n_dims, typename IdxLin = linear_index_type>
    RAJA_INLINE
    ShiftedView shift(const std::array<IdxLin, N>& shift)
    {
      static_assert(n_dims==layout_type::n_dims, "Dimension mismatch in view shift");

      shifted_layout_type shift_layout(get_layout());
      shift_layout.shift(shift);

      return ShiftedView(get_data(), shift_layout);
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
    auto operator()(Args... args) const ->
    typename internal::ViewVectorHelper<linear_index_type, value_type, pointer_type, typename TypedViewVectorHelper<IndexTypes, camp::decay<Args>>::type...>::type
    {
      return Base::operator_internal(vectorArgExtractor<IndexTypes>(args)...);
    }


    template <typename Arg>
    RAJA_HOST_DEVICE
    RAJA_INLINE
    auto operator[](Arg arg) const ->
    typename internal::ViewVectorHelper<linear_index_type, value_type, pointer_type, typename TypedViewVectorHelper<IndexTypes, camp::decay<Arg>>::type ...>::type
    {
      return Base::operator_internal(vectorArgExtractor<IndexTypes...>(arg));
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
