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

#include "RAJA/config.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/pattern/tensor/internal/VectorRegisterBase.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/Operators.hpp"

#ifdef RAJA_ENABLE_CUDA

#include "RAJA/policy/cuda/reduce.hpp"

#ifndef RAJA_policy_tensor_arch_cuda_cuda_warp_register_HPP
#define RAJA_policy_tensor_arch_cuda_cuda_warp_register_HPP



namespace RAJA {



  template<typename ELEMENT_TYPE>
  class TensorRegister<cuda_warp_register, ELEMENT_TYPE,
                       VectorLayout,
                       camp::idx_seq<32>>:
    public internal::VectorRegisterBase<TensorRegister<
                      cuda_warp_register,
                      ELEMENT_TYPE, VectorLayout,
                      camp::idx_seq<32> > >
  {
    public:
      using register_policy = cuda_warp_register;
      using self_type = TensorRegister<cuda_warp_register, ELEMENT_TYPE,
          VectorLayout,
          camp::idx_seq<32> >;
      using element_type = ELEMENT_TYPE;
      using register_type = ELEMENT_TYPE;

		private:
      element_type m_value;

		public:

      static constexpr int s_num_elem = 32;

      /*!
       * @brief Default constructor, zeros register contents
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      TensorRegister() : m_value(element_type(0)) {
      }

      /*!
       * @brief Copy constructor from raw value
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      explicit TensorRegister(element_type const &c) : m_value(c) {}


      /*!
       * @brief Copy constructor
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      TensorRegister(self_type const &c) : m_value(c.m_value) {}


      /*!
       * @brief Copy assignment operator
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type &operator=(self_type const &c){
        m_value = c.m_value;
        return *this;
      }

      /*!
       * @brief Gets our warp lane
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      static
      int get_lane() {
#ifdef __CUDA_ARCH__
        return threadIdx.x;
//        int lane;
//        //asm volatile ("mov.s32 %0, %laneid;" : "=r"(lane));
//        asm ("mov.s32 %0, %laneid;" : "=r"(lane));
//        return lane;
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
       * @brief Load a full register from a stride-one memory location
       *
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type &load_packed(element_type const *ptr){



        auto lane = get_lane();


//        printf("load_packed(lane=%d, %p)\n", (int)lane, ptr); return *this;

        m_value = ptr[lane];

        return *this;
      }

      /*!
       * @brief Partially load a register from a stride-one memory location given
       *        a run-time number of elements.
       *
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type &load_packed_n(element_type const *ptr, int N){
        auto lane = get_lane();
//        printf("load_packed_n(lane=%d, %p, n=%d)\n", (int)lane, ptr, N);return *this;
        if(lane < N){
          m_value = ptr[lane];
        }
        else{
          m_value = element_type(0);
        }
        return *this;
      }

      /*!
       * @brief Gather a full register from a strided memory location
       *
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type &load_strided(element_type const *ptr, int stride){

        auto lane = get_lane();

//        printf("load_strided(lane=%d, stride=%d, %p)\n", (int)lane, (int)stride, ptr);return *this;

        m_value = ptr[stride*lane];

        return *this;
      }


      /*!
       * @brief Partially load a register from a stride-one memory location given
       *        a run-time number of elements.
       *
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type &load_strided_n(element_type const *ptr, int stride, int N){
        auto lane = get_lane();

//        printf("load_strided_n(lane=%d, stride=%d, n=%d, %p)\n", (int)lane, (int)stride, N, ptr);return *this;

        if(lane < N){
          m_value = ptr[stride*lane];
        }
        else{
          m_value = element_type(0);
        }
        return *this;
      }


      /*!
       * @brief Store entire register to consecutive memory locations
       *
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type const &store_packed(element_type *ptr) const{

        auto lane = get_lane();

//        printf("store_packed(lane=%d, %p)\n", (int)lane, ptr);return *this;

        ptr[lane] = m_value;

        return *this;
      }

      /*!
       * @brief Store entire register to consecutive memory locations
       *
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type const &store_packed_n(element_type *ptr, int N) const{
        auto lane = get_lane();

//        printf("store_packed_n(lane=%d, %p, n=%d)\n", (int)lane, ptr, N);return *this;

        if(lane < N){
          ptr[lane] = m_value;
        }
        return *this;
      }

      /*!
       * @brief Store entire register to consecutive memory locations
       *
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type const &store_strided(element_type *ptr, int stride) const{

        auto lane = get_lane();

//        printf("store_strided(lane=%d, stride=%d, %p)\n", (int)lane, (int)stride, ptr);return *this;

        ptr[lane*stride] = m_value;

        return *this;
      }


      /*!
       * @brief Store partial register to consecutive memory locations
       *
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type const &store_strided_n(element_type *ptr, int stride, int N) const{


        auto lane = get_lane();

//        printf("store_strided_n(lane=%d, stride=%d, n=%d, %p)\n", (int)lane, (int)stride, N, ptr);return *this;


        if(lane < N){
          ptr[lane*stride] = m_value;
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
      RAJA_HOST_DEVICE
      element_type get(int i) const
			{
#ifdef __CUDA_ARCH__
        return  __shfl_sync(0xffffffff, m_value, i, 32);
#else
        return m_value;
#endif
			}

      /*!
       * @brief Set scalar value in vector register
       * @param i Offset of scalar to set
       * @param value Value of scalar to set
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type &set(int i, element_type value)
			{
#ifdef __CUDA_ARCH__
				auto lane = get_lane();
      	if(lane == i){
					m_value = value;
				}
#else
        m_value = value;
#endif
        return *this;
			}


      RAJA_HOST_DEVICE
      RAJA_HOST_DEVICE
      self_type &broadcast(element_type const &a){
        m_value = a;
        return *this;
      }

      /*!
       * @brief Extracts a scalar value and broadcasts to a new register
       */
      RAJA_HOST_DEVICE
      RAJA_HOST_DEVICE
      self_type get_and_broadcast(int i) const {
#ifdef __CUDA_ARCH__
        self_type x;
        x.m_value = __shfl_sync(0xffffffff, m_value, i, 32);
        return x;
#else
        return self_type(m_value);
#endif
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &copy(self_type const &src){
        m_value = src.m_value;
        return *this;
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type add(self_type const &b) const {
        return self_type(m_value + b.m_value);
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type add(element_type b) const {
        return self_type(m_value + b);
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type subtract(self_type const &b) const {
        return self_type(m_value - b.m_value);
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type multiply(self_type const &b) const {
        return self_type(m_value * b.m_value);
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type divide(self_type const &b) const {
        return self_type(m_value / b.m_value);
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type divide_n(self_type const &b, int N) const {
        return get_lane() < N ? self_type(m_value / b.m_value) : self_type(element_type(0));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type fused_multiply_add(self_type const &b, self_type const &c) const
      {
#ifdef __CUDA_ARCH__
        return self_type(fma(m_value, b.m_value, c.m_value));
#else
        return m_value*c.m_value + c.m_value;
#endif
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type fused_multiply_subtract(self_type const &b, self_type const &c) const
      {
#ifdef __CUDA_ARCH__
        return self_type(fma(m_value, b.m_value, -c.m_value));
#else
        return m_value*c.m_value - c.m_value;
#endif
      }


      /*!
       * @brief Sum the elements of this vector
       * @return Sum of the values of the vectors scalar elements
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      element_type sum(int N = s_num_elem) const
      {
#ifdef __CUDA_ARCH__
				// Allreduce sum
				using combiner_t = RAJA::reduce::detail::op_adapter<element_type, RAJA::operators::plus>;

				auto ident = element_type();
				auto lane = get_lane();
				auto value = lane < N ? m_value : ident;
				return RAJA::cuda::impl::partial_warp_allreduce<combiner_t, 5, element_type>(value);
#else
				return N > 0 ? m_value : element_type(0);
#endif
      }



      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      element_type max(int N = s_num_elem) const
      {
#ifdef __CUDA_ARCH__
        // Allreduce maximum
        using combiner_t = RAJA::reduce::detail::op_adapter<element_type, RAJA::operators::maximum>;

        auto ident = element_type();
        auto lane = get_lane();
        auto value = lane < N ? m_value : ident;
        return RAJA::cuda::impl::partial_warp_allreduce<combiner_t, 5, element_type>(value);
#else
        return N > 0 ? m_value : element_type();
#endif
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type vmax(self_type a) const
      {
        return self_type{RAJA::max(m_value, a.m_value)};
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      element_type min(int N = s_num_elem) const
      {
#ifdef __CUDA_ARCH__
        // Allreduce minimum
        using combiner_t = RAJA::reduce::detail::op_adapter<element_type, RAJA::operators::minimum>;

        auto ident = element_type();
        auto lane = get_lane();
        auto value = lane < N ? m_value : ident;
        return RAJA::cuda::impl::partial_warp_allreduce<combiner_t, 5, element_type>(value);
        return RAJA::cuda::impl::partial_warp_allreduce<combiner_t, 5, element_type>(value);
#else
        return N > 0 ? m_value : element_type();
#endif
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type vmin(self_type a) const
      {
        return self_type{RAJA::min(m_value, a.m_value)};
      }
  };




} // namespace RAJA


#endif // Guard

#endif // CUDA
