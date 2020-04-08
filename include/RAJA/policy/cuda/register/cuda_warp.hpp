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

  /*!
   * A warp distributed vector.
   */
	template<camp::idx_t LANE_BITS>
  struct cuda_warp_register {
	    static_assert(LANE_BITS >= 1 && LANE_BITS <= 5, "Invalid number of lanes");
	};



  template<camp::idx_t LANE_BITS, typename ELEMENT_TYPE>
  class Register<cuda_warp_register<LANE_BITS>, ELEMENT_TYPE> :
    public internal::RegisterBase<Register<cuda_warp_register<LANE_BITS>, ELEMENT_TYPE>>
  {
    public:
      using register_policy = cuda_warp_register<LANE_BITS>;
      using self_type = Register<cuda_warp_register<LANE_BITS>, ELEMENT_TYPE>;
      using element_type = ELEMENT_TYPE;
      using register_type = element_type;

      using bitmask_t = BitMask<LANE_BITS, 0>;

		private:
      element_type m_value;


		public:

      static constexpr camp::idx_t s_num_elem = 1<<(LANE_BITS);

      /*!
       * @brief Default constructor, zeros register contents
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      Register() : m_value(element_type(0)) {
      }

      /*!
       * @brief Copy constructor from raw value
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      explicit Register(element_type const &c) : m_value(c) {}


      /*!
       * @brief Copy constructor
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      Register(self_type const &c) : m_value(c.m_value) {}


      /*!
       * @brief Gets our lane after our bitmask has been applied
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      constexpr
      int get_lane() {
#ifdef __CUDA_ARCH__
        return bitmask_t::maskValue(threadIdx.x);
#else
        return 0;
#endif
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
      RAJA_HOST_DEVICE
      self_type &load(element_type const *ptr, camp::idx_t stride = 1, camp::idx_t N = s_num_elem){
        auto lane = get_lane();
        if(lane < s_num_elem){
          m_value = ptr[stride*lane];
        }
        return *this;
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
      RAJA_HOST_DEVICE
      self_type const &store(element_type *ptr, camp::idx_t stride = 1, camp::idx_t N = s_num_elem) const{
        auto lane = get_lane();
        if(lane < s_num_elem){
          ptr[stride*lane] = m_value;
        }
        return *this;
      }

      /*!
       * @brief Get scalar value from vector register
       * @param i Offset of scalar to get
       * @return Returns scalar value at i
       */
      constexpr
      RAJA_INLINE
      RAJA_DEVICE
      element_type get(camp::idx_t i) const
			{
				return __shfl_sync(0xffffffff, m_value, i);
			}

      /*!
       * @brief Set scalar value in vector register
       * @param i Offset of scalar to set
       * @param value Value of scalar to set
       */
      RAJA_INLINE
      RAJA_DEVICE
      self_type &set(camp::idx_t i, element_type value)
			{
				auto lane = get_lane();
      	if(lane == i){
					m_value = value;
				}
      	return *this;
			}


      RAJA_DEVICE
      RAJA_INLINE
      self_type &broadcast(element_type const &a){
        m_value = a;
        return *this;
      }

      RAJA_DEVICE
      RAJA_INLINE
      self_type &copy(self_type const &src){
        m_value = src.m_value;
        return *this;
      }

      RAJA_DEVICE
      RAJA_INLINE
      self_type add(self_type const &b) const {
        return self_type(m_value + b.m_value);
      }

      RAJA_DEVICE
      RAJA_INLINE
      self_type subtract(self_type const &b) const {
        return self_type(m_value - b.m_value);
      }

      RAJA_DEVICE
      RAJA_INLINE
      self_type multiply(self_type const &b) const {
        return self_type(m_value * b.m_value);
      }

      RAJA_DEVICE
      RAJA_INLINE
      self_type divide(self_type const &b, camp::idx_t N = s_num_elem) const {
        return get_lane() < N ? self_type(m_value / b.m_value) : self_type(element_type(0));
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
      element_type sum(camp::idx_t N = s_num_elem) const
      {
				// Allreduce sum
				using combiner_t = RAJA::reduce::detail::op_adapter<element_type, RAJA::operators::plus>;
			
				auto ident = element_type();
				auto lane = get_lane();
				auto value = lane < N ? m_value : ident;
				return RAJA::cuda::impl::partial_warp_allreduce<combiner_t, LANE_BITS, element_type>(value);
      }



      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      RAJA_DEVICE
      element_type max(camp::idx_t N = s_num_elem) const
      {
        // Allreduce maximum
        using combiner_t = RAJA::reduce::detail::op_adapter<element_type, RAJA::operators::maximum>;

        auto ident = element_type();
        auto lane = get_lane();
        auto value = lane < N ? m_value : ident;
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
      element_type min(camp::idx_t N = s_num_elem) const
      {
        // Allreduce minimum
        using combiner_t = RAJA::reduce::detail::op_adapter<element_type, RAJA::operators::minimum>;

        auto ident = element_type();
        auto lane = get_lane();
        auto value = lane < N ? m_value : ident;
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
