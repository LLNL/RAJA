/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing SIMT distrubuted register abstractions for
 *          CUDA
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include<RAJA/config.hpp>
#include "RAJA/util/macros.hpp"
#include "RAJA/util/BitMask.hpp"
#include "RAJA/util/Operators.hpp"

#ifdef RAJA_ENABLE_CUDA

#ifndef RAJA_policy_cuda_register_cuda_warp_HPP
#define RAJA_policy_cuda_register_cuda_warp_HPP



namespace RAJA {
	template<size_t LANE_BITS>
  struct vector_cuda_warp_register {};

  template<size_t LANE_BITS, typename T>
  struct RegisterTraits<vector_cuda_warp_register<LANE_BITS>, T>{

		static_assert(LANE_BITS >= 1 && LANE_BITS <= 5, "Invalid number of lanes");

      using register_type = T;
      using element_type = T;

      static constexpr size_t s_num_elem = 1 << (LANE_BITS);
      static constexpr size_t s_byte_width = s_num_elem * sizeof(T);
      static constexpr size_t s_bit_width = s_byte_width * 8;

  };




  template<size_t LANE_BITS, typename ELEMENT_TYPE, size_t NUM_ELEM>
  class Register<vector_cuda_warp_register<LANE_BITS>, ELEMENT_TYPE, NUM_ELEM> :
    public internal::RegisterBase<Register<vector_cuda_warp_register<LANE_BITS>, ELEMENT_TYPE, NUM_ELEM>>
  {
    public:
      using self_type = Register<vector_cuda_warp_register<LANE_BITS>, ELEMENT_TYPE, NUM_ELEM>;
      using element_type = ELEMENT_TYPE;
      using register_type = element_type;

      static constexpr size_t s_num_elem = NUM_ELEM;
      static constexpr size_t s_byte_width = s_num_elem*sizeof(ELEMENT_TYPE);
      static constexpr size_t s_bit_width = s_byte_width*8;

      using bitmask_t = BitMask<LANE_BITS, 0>;

			static_assert(s_num_elem <= RegisterTraits<vector_cuda_warp_register<LANE_BITS>, ELEMENT_TYPE>::s_num_elem, "Too many elements");

		private:
      element_type m_value;


		public:

      /*!
       * @brief Default constructor, zeros register contents
       */
      RAJA_INLINE
      RAJA_DEVICE
      Register() : m_value(element_type(0)) {
      }

      /*!
       * @brief Copy constructor from raw value
       */
      RAJA_INLINE
      RAJA_DEVICE
      constexpr
      explicit Register(element_type const &c) : m_value(c) {}


      /*!
       * @brief Copy constructor
       */
      RAJA_INLINE
      RAJA_DEVICE
      constexpr
      Register(self_type const &c) : m_value(c.m_value) {}


      /*!
       * @brief Gets our lane after our bitmask has been applied
       */
      RAJA_INLINE
      RAJA_DEVICE
      static
      constexpr
      int get_lane() {
        return bitmask_t::maskValue(threadIdx.x);
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr
      bool is_root() {
        return get_lane() == 0;
      }


      /*!
       * @brief Strided load constructor, when scalars are located in memory
       * locations ptr, ptr+stride, ptr+2*stride, etc.
       *
       *
       * Note: this could be done with "gather" instructions if they are
       * available. (like in avx2, but not in avx)
       */
      RAJA_INLINE
      RAJA_DEVICE
      void load(element_type const *ptr, size_t stride = 1){
        auto lane = get_lane();
        if(lane < s_num_elem){
          m_value = ptr[stride*lane];
        }
      }


      /*!
       * @brief Strided store operation, where scalars are stored in memory
       * locations ptr, ptr+stride, ptr+2*stride, etc.
       *
       *
       * Note: this could be done with "scatter" instructions if they are
       * available.
       */
      RAJA_INLINE
      RAJA_DEVICE
      void store(element_type *ptr, size_t stride) const{
        auto lane = get_lane();
        if(lane < s_num_elem){
          ptr[stride*lane] = m_value;
        }
      }

      /*!
       * @brief Get scalar value from vector register
       * @param i Offset of scalar to get
       * @return Returns scalar value at i
       */
      template<typename IDX>
      constexpr
      RAJA_INLINE
      RAJA_DEVICE
      element_type get(IDX i) const
			{
				return __shfl_sync(0xffffffff, m_value, i);
			}

      /*!
       * @brief Set scalar value in vector register
       * @param i Offset of scalar to set
       * @param value Value of scalar to set
       */
      template<typename IDX>
      RAJA_INLINE
      RAJA_DEVICE
      void set(IDX i, element_type value)
			{
				auto lane = get_lane();
      	if(lane == i){
					m_value = value;
				}
			}


      RAJA_DEVICE
      RAJA_INLINE
      static
      self_type broadcast(element_type const &a){
        return self_type(a);
      }

      RAJA_DEVICE
      RAJA_INLINE
      static
      void copy(self_type &dst, self_type const &src){
        dst.m_value = src.m_value;
      }

      RAJA_DEVICE
      RAJA_INLINE
      constexpr
      self_type add(self_type const &b) const {
        return self_type(m_value + b.m_value);
      }

      RAJA_DEVICE
      RAJA_INLINE
      constexpr
      self_type subtract(self_type const &b) const {
        return self_type(m_value - b.m_value);
      }

      RAJA_DEVICE
      RAJA_INLINE
      constexpr
      self_type multiply(self_type const &b) const {
        return self_type(m_value * b.m_value);
      }

      RAJA_DEVICE
      RAJA_INLINE
      constexpr
      self_type divide(self_type const &b) const {
        return self_type(m_value / b.m_value);
      }

      RAJA_DEVICE
      RAJA_INLINE
      self_type fused_multiply_add(self_type const &b, self_type const &c) const
      {
        return self_type(fma(m_value, b.m_value, c.m_value));
      }

      RAJA_DEVICE
      RAJA_INLINE
      self_type fused_multiply_subtract(self_type const &b, self_type const &c) const
      {
        return self_type(fma(m_value, b.m_value, -c.m_value));
      }


      /*!
       * @brief Sum the elements of this vector
       * @return Sum of the values of the vectors scalar elements
       */
      RAJA_INLINE
      RAJA_DEVICE
      element_type sum() const
      {
				// Allreduce sum
				using combiner_t = RAJA::reduce::detail::op_adapter<element_type, RAJA::operators::plus>;
			
				auto ident = element_type();
				auto lane = get_lane();
				auto value = lane < s_num_elem ? m_value : ident;
				return RAJA::cuda::impl::partial_warp_allreduce<combiner_t, LANE_BITS, element_type>(value);
      }



      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      RAJA_DEVICE
      element_type max() const
      {
        // Allreduce maximum
        using combiner_t = RAJA::reduce::detail::op_adapter<element_type, RAJA::operators::maximum>;

        auto ident = element_type();
        auto lane = get_lane();
        auto value = lane < s_num_elem ? m_value : ident;
        return RAJA::cuda::impl::partial_warp_allreduce<combiner_t, LANE_BITS, element_type>(value);
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_INLINE
      RAJA_DEVICE
      self_type vmax(self_type a) const
      {
        return self_type{RAJA::max(m_value, a.m_value)};
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      RAJA_DEVICE
      element_type min() const
      {
        // Allreduce minimum
        using combiner_t = RAJA::reduce::detail::op_adapter<element_type, RAJA::operators::minimum>;

        auto ident = element_type();
        auto lane = get_lane();
        auto value = lane < s_num_elem ? m_value : ident;
        return RAJA::cuda::impl::partial_warp_allreduce<combiner_t, LANE_BITS, element_type>(value);
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_INLINE
      RAJA_DEVICE
      self_type vmin(self_type a) const
      {
        return self_type{RAJA::min(m_value, a.m_value)};
      }
  };




} // namespace RAJA


#endif // Guard

#endif // CUDA
