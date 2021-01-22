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

#ifndef RAJA_pattern_tensor_TensorBlock_HPP
#define RAJA_pattern_tensor_TensorBlock_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "camp/camp.hpp"
#include "RAJA/pattern/tensor/TensorRegister.hpp"
#include "RAJA/util/StaticLayout.hpp"

namespace RAJA
{

  template<camp::idx_t ... IDX>
  struct GetSeqValue;

  template<camp::idx_t IDX0, camp::idx_t ... IDX_REST>
  struct GetSeqValue<IDX0, IDX_REST...>
  {
      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr
      camp::idx_t get(camp::idx_t idx){
        return idx == 0 ? IDX0 : GetSeqValue<IDX_REST...>::get(idx-1);
      }
  };
  template<>
    struct GetSeqValue<>
    {
        RAJA_HOST_DEVICE
        RAJA_INLINE
        static
        constexpr
        camp::idx_t get(camp::idx_t ){
          return 0;
        }
    };

  template <typename IdxLin,
            IdxLin... RangeInts,
            IdxLin... Sizes,
            IdxLin... Strides>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  constexpr
  RAJA::internal::TensorTile<IdxLin, RAJA::internal::TENSOR_FULL, sizeof...(RangeInts)>
  get_layout_default_tile(
      RAJA::detail::StaticLayoutBase_impl<IdxLin,
      camp::int_seq<IdxLin, RangeInts...>,
      camp::int_seq<IdxLin, Sizes...>,
      camp::int_seq<IdxLin, Strides...>> const &)
  {
    using tile_type = RAJA::internal::TensorTile<IdxLin, RAJA::internal::TENSOR_FULL, sizeof...(RangeInts)>;
    return tile_type{ {IdxLin(0*Sizes)...}, {IdxLin(Sizes)...}};
  }


  template<typename STORAGE_POLICY, typename T,  typename LAYOUT>
  class TensorBlockStorage : public RAJA::internal::ViewBase<T, T*, LAYOUT>
  {
    public:
      using self_type = TensorBlockStorage<STORAGE_POLICY, T,  LAYOUT>;
      using base_type = RAJA::internal::ViewBase<T, T*, LAYOUT>;
      using element_type = T;
      using layout_type = LAYOUT;
      //using index_type = typename LAYOUT::IndexLinear;
      static constexpr camp::idx_t s_num_elem = LAYOUT::s_size;
      using tile_type = decltype(get_layout_default_tile(layout_type{}));

      //using ref_type = internal::TensorRef<ElementType*, LinIdx, internal::TENSOR_MULTIPLE, s_num_dims, s_stride_one_dim>;


      /*!
       * Gets the default tile of this storage block
       * That tile always start at 0, and extends to the full tile sizes
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr
      decltype(get_layout_default_tile(layout_type{}))
      s_get_default_tile()
      {
        return get_layout_default_tile(layout_type{});
      }


      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      TensorBlockStorage() noexcept :
        base_type(&m_data[0], layout_type{})
      {
      }


      template<typename TILE_TYPE, camp::idx_t ... DIM_SEQ>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      auto create_ref_expanded(TILE_TYPE const &tile, camp::idx_seq<DIM_SEQ...> const &)  noexcept ->
        internal::TensorRef<T*, typename TILE_TYPE::index_type, TILE_TYPE::s_tensor_size, layout_type::n_dims, layout_type::stride_one_dim>
      {
        using ref_t =
            internal::TensorRef<T*, typename TILE_TYPE::index_type, TILE_TYPE::s_tensor_size, layout_type::n_dims, layout_type::stride_one_dim>;

        return ref_t{&m_data[0], {(typename TILE_TYPE::index_type)base_type::get_layout().template get_dim_stride<DIM_SEQ>()...}, tile};
      }

      template<typename TILE_TYPE>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      auto create_ref(TILE_TYPE const &tile) noexcept ->
        decltype(create_ref_expanded(tile, camp::make_idx_seq_t<layout_type::n_dims>{}))
      {
        return create_ref_expanded(tile, camp::make_idx_seq_t<layout_type::n_dims>{});
      }

      template<typename TILE_TYPE, camp::idx_t ... DIM_SEQ>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      auto create_ref_expanded(TILE_TYPE const &tile, camp::idx_seq<DIM_SEQ...> const &) const noexcept ->
        internal::TensorRef<T const*, typename TILE_TYPE::index_type, TILE_TYPE::s_tensor_size, layout_type::n_dims, layout_type::stride_one_dim>
      {
        using ref_t =
            internal::TensorRef<T const*, typename TILE_TYPE::index_type, TILE_TYPE::s_tensor_size, layout_type::n_dims, layout_type::stride_one_dim>;

        return ref_t{&m_data[0], {(typename TILE_TYPE::index_type)base_type::get_layout().template get_dim_stride<DIM_SEQ>()...}, tile};
      }

      template<typename TILE_TYPE>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      auto create_ref(TILE_TYPE const &tile) const noexcept ->
        decltype(create_ref_expanded(tile, camp::make_idx_seq_t<layout_type::n_dims>{}))
      {
        return create_ref_expanded(tile, camp::make_idx_seq_t<layout_type::n_dims>{});
      }


//      RAJA_HOST_DEVICE
//      RAJA_INLINE
//      constexpr
//      self_type createTemporary() const noexcept{
//        return self_type{};
//      }

    private:
      T m_data[s_num_elem];
  };


//  namespace internal
//  {
//
//    template<typename STORAGE, typename REGISTER_TYPE, typename REF_TYPE>
//    struct TensorBlockLoadFunctor
//    {
//        STORAGE &m_storage;
//        REF_TYPE const &m_memory_ref;
//
//        template<typename TILE_TYPE>
//        RAJA_HOST_DEVICE
//        RAJA_INLINE
//        void operator()(TILE_TYPE const &memory_tile) const{
//
//          // Create a register and perform the load
//          REGISTER_TYPE r;
//          auto lref = merge_ref_tile(m_memory_ref, memory_tile);
//          r.load_ref(lref);
//
//          // Compute the block tile by subtracting off the memory tile's
//          // starting position
//          auto block_tile = memory_tile - m_memory_ref.m_tile;
//
//          // Compute a ref into storage, and have the register perform the
//          // store
//          auto sref = m_storage.create_ref(block_tile);
//          r.store_ref(sref);
//
////          printf("LoadFunctor: size=%d\n", (int)STORAGE::s_num_elem);
////          printf("lref:\n"); lref.print();
////          printf("sref:\n"); sref.print();
//
//
//        }
//    };
//
//
//    template<typename STORAGE, typename REGISTER_TYPE, typename REF_TYPE>
//    struct TensorBlockStoreFunctor
//    {
//        STORAGE const &m_storage;
//        REF_TYPE const &m_memory_ref;
//
//        template<typename TILE_TYPE>
//        RAJA_HOST_DEVICE
//        RAJA_INLINE
//        void operator()(TILE_TYPE const &memory_tile) const{
//          // Compute the block tile by subtracting off the memory tile's
//          // starting position
//          auto block_tile = memory_tile - m_memory_ref.m_tile;
//
//          // Create a register and perform the load from the block
//          REGISTER_TYPE r;
//          auto lref = m_storage.create_ref(block_tile);
//          r.load_ref(lref);
//
//          // Store into memory
//          auto sref = merge_ref_tile(m_memory_ref, memory_tile);
//          r.store_ref(sref);
//
//
////          printf("StoreFunctor: size=%d\n", (int)STORAGE::s_num_elem);
////          printf("lref:\n"); lref.print();
////          printf("sref:\n"); sref.print();
//        }
//    };
//
//
//    template<typename STORAGE, typename REGISTER_TYPE>
//    struct TensorBlockBroadcastFunctor
//    {
//        STORAGE &m_storage;
//        REGISTER_TYPE m_reg_value;
//
//        template<typename TILE_TYPE>
//        RAJA_HOST_DEVICE
//        RAJA_INLINE
//        void operator()(TILE_TYPE const &block_tile) const{
//          // store the register at the specified tile
//          m_reg_value.store_ref(m_storage.create_ref(block_tile));
//        }
//    };
//
//    template<typename STOR_RESULT, typename STOR_RIGHT, typename REGISTER_TYPE>
//    struct TensorBlockAddFunctor
//    {
//        STOR_RESULT &m_result;
//        STOR_RIGHT const &m_right;
//
//        template<typename TILE_TYPE>
//        RAJA_HOST_DEVICE
//        RAJA_INLINE
//        void operator()(TILE_TYPE const &block_tile) const{
//          // load the left and right operands
//          REGISTER_TYPE left, right;
//          left.load_ref(m_result.create_ref(block_tile));
//          right.load_ref(m_right.create_ref(block_tile));
//
//          // compute result and store
//          left.add(right).store_ref(m_result.create_ref(block_tile));
//        }
//    };
//
//    template<typename ACC, typename LEFT, typename RIGHT, typename REGISTER_TYPE>
//    struct TensorBlockMatrixMultiplyFunctor
//    {
//        ACC &m_acc;
//        LEFT const &m_left;
//        RIGHT const &m_right;
//
//        using multiply_op = internal::ET::MultiplyOperator<LEFT, RIGHT>;
//
//        template<typename TILE_TYPE>
//        RAJA_HOST_DEVICE
//        RAJA_INLINE
//        void operator()(TILE_TYPE const &block_tile) const{
//
//          // do multiplication on this tile
//          multiply_op::multiply(m_acc, block_tile, m_left, m_right);
//
//        }
//    };
//
//  } // namespace internal




  template<typename REGISTER_POLICY,
           typename ELEMENT_TYPE,
           typename LAYOUT,
           typename SIZES,
           typename STORAGE_POLICY>
  class TensorBlock;





  template<typename REGISTER_POLICY,
           typename ELEMENT_TYPE,
           camp::idx_t ... LAYOUT,
           camp::idx_t ... DIM_SIZES,
           typename STORAGE_POLICY>

  class TensorBlock<REGISTER_POLICY,
                    ELEMENT_TYPE,
                    camp::idx_seq<LAYOUT...>,
                    camp::idx_seq<DIM_SIZES...>,
                    STORAGE_POLICY>
  {


    public:
      using self_type = TensorBlock<REGISTER_POLICY, ELEMENT_TYPE,
          camp::idx_seq<LAYOUT...>,
          camp::idx_seq<DIM_SIZES...>,
          STORAGE_POLICY>;
      using element_type = camp::decay<ELEMENT_TYPE>;

      using register_type =
            TensorRegister<REGISTER_POLICY,
                           ELEMENT_TYPE,
                           //TensorLayout<LAYOUT...>,
                           RowMajorLayout,
                           camp::idx_seq<
                              (0*LAYOUT+RegisterTraits<REGISTER_POLICY,ELEMENT_TYPE>::s_num_elem)...>,
                           camp::make_idx_seq_t<RegisterTraits<REGISTER_POLICY,ELEMENT_TYPE>::s_num_elem>>;

      static constexpr camp::idx_t s_num_dims = sizeof...(DIM_SIZES);

      using storage_layout = RAJA::StaticLayout<camp::idx_seq<LAYOUT...>, DIM_SIZES...>;
      using storage_type = TensorBlockStorage<STORAGE_POLICY, ELEMENT_TYPE, storage_layout>;

      using index_type = camp::idx_t;

      using tile_type = typename storage_type::tile_type;
      using ref_type = internal::TensorRef<element_type *, typename tile_type::index_type, tile_type::s_tensor_size, storage_layout::n_dims, storage_layout::stride_one_dim>;

      // to masquerade as a ET node
      using result_type = register_type;

    private:
      storage_type m_storage;


      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type *getThis(){
        return static_cast<self_type *>(this);
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      self_type const *getThis() const{
        return static_cast<self_type const *>(this);
      }

    public:

      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr
      bool is_root() {
        return true;
      }



      RAJA_HOST_DEVICE
      RAJA_INLINE
      storage_type &get_storage() {
        return m_storage;
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      storage_type const &get_storage() const {
        return m_storage;
      }

      /*!
       * Gets the size of the tensor
       * Since this is a vector, just the length of the vector in dim 0
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr int s_dim_elem(int dim){
        return GetSeqValue<DIM_SIZES...>::get(dim);
      }

      /*!
       * Gets the size of the tensor
       * Since this is a vector, just the length of the vector in dim 0
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr int getDimSize(int dim) const{
        return GetSeqValue<DIM_SIZES...>::get(dim);
      }

      /*!
       * Gets the default tile of this tensor
       * That tile always start at 0, and extends to the full tile sizes
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      decltype(get_layout_default_tile(storage_layout{}))
      get_default_tile() const
      {
        return get_layout_default_tile(storage_layout{});
      }


      RAJA_HOST_DEVICE
      RAJA_INLINE
      auto
      get_ref() ->
      decltype(m_storage.create_ref(get_default_tile()))
      {
        return m_storage.create_ref(get_default_tile());
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      auto
      get_ref() const ->
      decltype(m_storage.create_ref(get_default_tile()))
      {
        return m_storage.create_ref(get_default_tile());
      }


      template<typename REF_TYPE>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      internal::ET::TensorLoadStore<register_type, REF_TYPE>
      create_et_store_ref(REF_TYPE const &ref) {
        return internal::ET::TensorLoadStore<register_type, REF_TYPE>(ref);
      }

      template<typename REF_TYPE>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      internal::ET::TensorLoadStore<register_type, REF_TYPE>
      s_load_ref(REF_TYPE const &ref) {
        return internal::ET::TensorLoadStore<register_type, REF_TYPE>(ref);
      }

//      /*!
//       * @brief convenience routine to allow Vector classes to use
//       * camp::sink() across a variety of register types, and use things like
//       * ternary operators
//       */
//      RAJA_HOST_DEVICE
//      RAJA_INLINE
//      constexpr
//      bool sink() const{
//        return false;
//      }


      TensorBlock() = default;
      ~TensorBlock() = default;

      TensorBlock(TensorBlock const &) = default;

      self_type &operator=(self_type const &x) = default;

      /*!
       * @brief Acts like a ET node, returning a member of the block
       */
      template<typename TILE_TYPE>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      internal::ET::TensorLoadStore<register_type, ref_type>
      eval(TILE_TYPE const &tile) const {
        //result.load_ref(m_storage.create_ref(tile));

        auto ref_left = getThis()->get_ref();
        internal::ET::TensorLoadStore<register_type, ref_type> et_left(ref_left);

        return et_left;
      }

//      /*!
//       * @brief Performs load specified by TensorRef object.
//       */
//      template<typename REF_TYPE>
//      RAJA_HOST_DEVICE
//      RAJA_INLINE
//      internal::ET::TensorLoadStore<register_type, REF_TYPE>
//      load_ref(REF_TYPE const &ref) const {
//
////        printf("TensorBlock: load_ref: "); ref.print(); printf("\n");
//
////        internal::TensorBlockLoadFunctor<storage_type, register_type, REF_TYPE>
////          functor{m_storage, ref};
////
////        internal::tensorTileExec<register_type>(ref.m_tile, functor);
////
////        return *getThis();
//
//        auto ref_left = getThis()->get_ref();
//        internal::ET::TensorLoadStore<register_type, REF_TYPE> et_left(ref_left);
//
//        return et_left;
//      }


//      /*!
//       * @brief Performs load specified by TensorRef object.
//       */
//      template<typename REF_TYPE>
//      RAJA_HOST_DEVICE
//      RAJA_INLINE
//      self_type const &store_ref(REF_TYPE const &ref) const{
//
////        printf("TensorBlock: store_ref: "); ref.print(); printf("\n");
//
//        internal::TensorBlockStoreFunctor<storage_type, register_type, REF_TYPE>
//          functor{m_storage, ref};
//
//        internal::tensorTileExec<register_type>(ref.m_tile, functor);
//
//        return *getThis();
//      }


//      template<typename ... IDX>
//      RAJA_INLINE
//      RAJA_HOST_DEVICE
//      element_type get(IDX const &... idx) const {
//        return m_storage(idx...);
//      }
//
//      template<typename ... IDX>
//      RAJA_INLINE
//      RAJA_HOST_DEVICE
//      void set(element_type const &value, IDX const &... idx) {
//        m_storage(idx...) = value;
//      }
//
//
//      /*!
//       * @brief Performs load specified by TensorRef object.
//       */
//      RAJA_HOST_DEVICE
//      RAJA_INLINE
//      self_type &broadcast(element_type value) {
//
//        auto ref_left = getThis()->get_ref();
//        internal::ET::TensorLoadStore<register_type, decltype(ref_left)> et_left(ref_left);
//
//        et_left = value;
//
//        return *getThis();
//      }
//
//
//      /*!
//       * In-place add operation
//       */
//      RAJA_INLINE
//      RAJA_HOST_DEVICE
//      self_type &inplace_add(self_type x){
//
//        internal::TensorBlockAddFunctor<storage_type, storage_type, register_type>
//          functor{m_storage, x.get_storage()};
//
//        internal::tensorTileExec<register_type>(storage_type::s_get_default_tile(), functor);
//
//        return *getThis();
//      }
//
//      /*!
//       * In-place sbutract operation
//       */
//      RAJA_INLINE
//      RAJA_HOST_DEVICE
//      self_type &inplace_subtract(self_type x){
//        *getThis() = getThis()->subtract(x);
//        return *getThis();
//      }
//
//      /*!
//       * In-place multiply operation
//       */
//      RAJA_INLINE
//      RAJA_HOST_DEVICE
//      self_type &inplace_multiply(self_type x){
//        *getThis() = getThis()->multiply(x);
//        return *getThis();
//      }
//
//      /*!
//       * In-place multiply-add operation
//       */
//      RAJA_INLINE
//      RAJA_HOST_DEVICE
//      self_type &inplace_multiply_add(self_type x, self_type y){
//        *getThis() = getThis()->multiply_add(x,y);
//        return *getThis();
//      }
//
//      /*!
//       * In-place multiply-subtract operation
//       */
//      RAJA_INLINE
//      RAJA_HOST_DEVICE
//      self_type &inplace_multiply_subtract(self_type x, self_type y){
//        *getThis() = getThis()->multiply_subtract(x,y);
//        return *getThis();
//      }
//
//      /*!
//       * In-place divide operation
//       */
//      RAJA_INLINE
//      RAJA_HOST_DEVICE
//      self_type &inplace_divide(self_type x){
//        *getThis() = getThis()->divide(x);
//        return *getThis();
//      }
//
//      /*!
//       * In-place scaling operation
//       */
//      RAJA_INLINE
//      RAJA_HOST_DEVICE
//      self_type &inplace_scale(element_type x){
//        *getThis() = getThis()->scale(x);
//        return *getThis();
//      }
//
//      /*!
//       * Matrix-Matrix multiply accumulate
//       */
//      template<typename ACCMAT, typename RMAT>
//      RAJA_HOST_DEVICE
//      RAJA_INLINE
//      void
//      matrix_multiply_accumulate(ACCMAT &acc, RMAT const &right) const {
//
//        auto ref_acc = acc.get_ref();
//        internal::ET::TensorLoadStore<register_type, decltype(ref_acc)> et_acc(ref_acc);
//
//        auto ref_left = getThis()->get_ref();
//        internal::ET::TensorLoadStore<register_type, decltype(ref_left)> et_left(ref_left);
//
//        auto ref_right = right.get_ref();
//        internal::ET::TensorLoadStore<register_type, decltype(ref_right)> et_right(ref_right);
//
//        et_acc = et_left * et_right + et_acc;
//
//      }

  };

}  // namespace RAJA



#endif
