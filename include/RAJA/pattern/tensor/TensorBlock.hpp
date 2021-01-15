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


  template<typename STORAGE_POLICY, typename T,  typename LAYOUT>
  class TensorBlockStorage : public RAJA::internal::ViewBase<T, T*, LAYOUT>
  {
    public:
      using self_type = TensorBlockStorage<STORAGE_POLICY, T,  LAYOUT>;
      using base_type = RAJA::internal::ViewBase<T, T*, LAYOUT>;
      using element_type = T;
      using layout_type = LAYOUT;

      static constexpr camp::idx_t s_num_elem = LAYOUT::s_size;

      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      TensorBlockStorage() noexcept :
        base_type(&m_data[0], layout_type{})
      {
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


  namespace internal
  {

    template<typename STORAGE, typename RHS_TYPE, typename REF_TYPE>
    struct TensorBlockLoadFunctor
    {
        RHS_TYPE const &rhs;
        REF_TYPE const &ref;

        template<typename TILE_TYPE>
        RAJA_HOST_DEVICE
        RAJA_INLINE
        void operator()(TILE_TYPE const &tile) const{

          // Create top-level storage
          STORAGE storage;

          // Call rhs to evaluate this tile
          rhs.eval(storage, tile);

          // Store result
          storage.store_ref(merge_ref_tile(ref, tile));
        }
    };

    template<typename STORAGE, typename RHS_TYPE, typename REF_TYPE>
    RAJA_HOST_DEVICE
    RAJA_INLINE
    constexpr
    auto makeTensorBlockLoadFunctor(RHS_TYPE const &rhs, REF_TYPE const &ref) ->
    TensorBlockLoadFunctor<STORAGE, RHS_TYPE, REF_TYPE>
    {
      return TensorBlockLoadFunctor<STORAGE, RHS_TYPE, REF_TYPE>{rhs, ref};
    }

  } // namespace internal




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
                           TensorLayout<LAYOUT...>,
                           camp::idx_seq<RegisterTraits<REGISTER_POLICY,ELEMENT_TYPE>::s_num_elem>,
                           camp::make_idx_seq_t<RegisterTraits<REGISTER_POLICY,ELEMENT_TYPE>::s_num_elem>>;

      static constexpr camp::idx_t s_num_dims = sizeof...(DIM_SIZES);

      using storage_layout = RAJA::StaticLayout<camp::idx_seq<LAYOUT...>, DIM_SIZES...>;
      using storage_type = TensorBlockStorage<STORAGE_POLICY, ELEMENT_TYPE, storage_layout>;

      using index_type = camp::idx_t;

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
       * Gets the default tile of this tensor
       * That tile always start at 0, and extends to the full tile sizes
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr internal::TensorTile<int, internal::TENSOR_FULL, s_num_dims>
      s_get_default_tile()
      {
        return internal::TensorTile<int, internal::TENSOR_FULL, s_num_dims>{
          {int(DIM_SIZES*0)...},
          {int(DIM_SIZES)...}
        };
      }

      /*!
       * @brief convenience routine to allow Vector classes to use
       * camp::sink() across a variety of register types, and use things like
       * ternary operators
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      bool sink() const{
        return false;
      }


      TensorBlock() = default;
      ~TensorBlock() = default;

      TensorBlock(TensorBlock const &) = default;

      self_type &operator=(self_type const &x) = default;

      /*!
       * @brief Get a TensorRef to this blocks data
       */


      /*!
       * @brief Performs load specified by TensorRef object.
       */
      template<typename REF_TYPE>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &load_ref(REF_TYPE const &ref){
//        internal::tensorTileExec<register_type>(ref.m_tile,
//                      internal::ET::makeTensorStoreFunctor<register_type>(*this, ref));
        return *getThis();
      }


      /*!
       * @brief Performs load specified by TensorRef object.
       */
      template<typename REF_TYPE>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type const &store_ref(REF_TYPE const &ref) const{
//        internal::tensorTileExec<register_type>(ref.m_tile,
//                      internal::ET::makeTensorStoreFunctor<register_type>(*this, ref));
        return *getThis();
      }


      template<typename ... IDX>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      element_type get(IDX const &... idx) const {
        return m_storage(idx...);
      }

      template<typename ... IDX>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      void set(element_type const &value, IDX const &... idx) {
        m_storage(idx...) = value;
      }


      /*!
       * In-place add operation
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type &inplace_add(self_type x){
        *getThis() = getThis()->add(x);
        return *getThis();
      }

      /*!
       * In-place sbutract operation
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type &inplace_subtract(self_type x){
        *getThis() = getThis()->subtract(x);
        return *getThis();
      }

      /*!
       * In-place multiply operation
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type &inplace_multiply(self_type x){
        *getThis() = getThis()->multiply(x);
        return *getThis();
      }

      /*!
       * In-place multiply-add operation
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type &inplace_multiply_add(self_type x, self_type y){
        *getThis() = getThis()->multiply_add(x,y);
        return *getThis();
      }

      /*!
       * In-place multiply-subtract operation
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type &inplace_multiply_subtract(self_type x, self_type y){
        *getThis() = getThis()->multiply_subtract(x,y);
        return *getThis();
      }

      /*!
       * In-place divide operation
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type &inplace_divide(self_type x){
        *getThis() = getThis()->divide(x);
        return *getThis();
      }

      /*!
       * In-place scaling operation
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type &inplace_scale(element_type x){
        *getThis() = getThis()->scale(x);
        return *getThis();
      }

  };

}  // namespace RAJA



#endif
