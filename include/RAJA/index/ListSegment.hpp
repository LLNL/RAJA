/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining list segment classes.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_ListSegment_HPP
#define RAJA_ListSegment_HPP

#include "RAJA/config.hpp"

#include <memory>
#include <type_traits>
#include <utility>

#include "camp/resource.hpp"

#include "RAJA/util/concepts.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/Span.hpp"
#include "RAJA/util/types.hpp"

#if defined(RAJA_CUDA_ACTIVE)
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"
#else
#define cudaErrchk(...)
#endif

#if defined(RAJA_ENABLE_HIP)
#include "RAJA/policy/hip/raja_hiperrchk.hpp"
#else
#define hipErrchk(...)
#endif

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  Class representing an arbitrary collection of indices.
 *
 *         Length indicates number of indices in index array.
 *         Traversal executes as:
 *            for (i = 0; i < getLength(); ++i) {
 *               expression using m_indx[i] as array index.
 *            }
 *
 ******************************************************************************
 */
template <typename T>
class TypedListSegment
{
/*
 * All of the following down to the 'public' section is original machinery 
 * to manage segment index data using CUDA or HIP unified memory. Eventually,
 * it will be removed, but is left in place for now to preserve original
 * behavior so our tests don't need to be reworked en masse now and users
 * won't see any different usage or behavior.
 */
  
#if defined(RAJA_DEVICE_ACTIVE)
  static constexpr bool Has_GPU = true;
#else
  static constexpr bool Has_GPU = false;
#endif

  //! tag for trivial per-element copy
  struct TrivialCopy {
  };
  //! tag for memcpy-style copy
  struct BlockCopy {
  };

  //! alias for GPU memory tag
  using GPU_memory = std::integral_constant<bool, true>;
  //! alias for CPU memory tag
  using CPU_memory = std::integral_constant<bool, false>;

  //! specialization for deallocation of GPU_memory
  void deallocate(GPU_memory) {
#if defined(RAJA_ENABLE_CUDA)
    cudaErrchk(cudaFree(m_data));
#elif defined(RAJA_ENABLE_HIP)
    hipErrchk(hipHostFree(m_data));
#endif
  }

  //! specialization for allocation of GPU_memory
  void allocate(GPU_memory)
  {
#if defined(RAJA_ENABLE_CUDA)
    cudaErrchk(cudaMallocManaged((void**)&m_data,
                                 m_size * sizeof(value_type),
                                 cudaMemAttachGlobal));
#elif defined(RAJA_ENABLE_HIP)
    hipErrchk(hipHostMalloc((void**)&m_data,
                            m_size * sizeof(value_type),
                            hipHostMallocMapped));
#endif
  }

  //! specialization for deallocation of CPU_memory
  void deallocate(CPU_memory) { delete[] m_data; }

  //! specialization for allocation of CPU_memory
  void allocate(CPU_memory) { m_data = new T[m_size]; }

#if defined(RAJA_CUDA_ACTIVE)
  //! copy data from container using BlockCopy
  template <typename Container>
  void copy(Container&& src, BlockCopy)
  {
    cudaErrchk(cudaMemcpy(
        m_data, &(*src.begin()), m_size * sizeof(T), cudaMemcpyDefault));
  }

#elif defined(RAJA_ENABLE_HIP)
  //! copy data from container using BlockCopy
  template <typename Container>
  void copy(Container&& src, BlockCopy)
  {
    memcpy(m_data, &(*src.begin()), m_size * sizeof(T));
  }
#endif

  //! copy data from container using TrivialCopy
  template <typename Container>
  void copy(Container&& source, TrivialCopy)
  {
    auto dest = m_data;
    auto src = source.begin();
    auto const end = source.end();
    while (src != end) {
      *dest = *src;
      ++dest;
      ++src;
    }
  }

  // internal helper to allocate data and populate with data from a container
  template <bool GPU, typename Container>
  void allocate_and_copy(Container&& src)
  {
    allocate(std::integral_constant<bool, GPU>());
    static constexpr bool use_gpu =
        GPU && std::is_pointer<decltype(src.begin())>::value &&
        std::is_same<type_traits::IterableValue<Container>, value_type>::value;
    using TagType =
        typename std::conditional<use_gpu, BlockCopy, TrivialCopy>::type;
    copy(src, TagType());
  }

public:
  //! value type for storage
  using value_type = T;

  //! iterator type for storage (will be a pointer)
  using iterator = T*;

  //! expose underlying index type
  using IndexType = RAJA::Index_type;

  //! prevent compiler from providing a default constructor
  TypedListSegment() = delete;

/*
 * The following two constructors allow users to specify a camp resource
 * for each list segment, which be used to manage segment index data.
 *
 * Eventually, I think it would be better to add a template parameter for
 * this class to specify the camp resource type rather than passing in a 
 * resource object.
 */

  ///
  /// \brief Construct list segment from given array with specified length
  ///        and use given camp resource to allocate list segment index data
  ///        if owned by this list segment.
  ///
  /// By default the ctor performs a deep copy of array elements.
  ///
  /// If 'Unowned' is passed as last argument, the constructed object
  /// does not own the segment data and will hold a pointer to given
  /// array's data. In this case, caller must manage object lifetimes properly.
  ///
  TypedListSegment(const value_type* values,
                   Index_type length,
                   camp::resources::Resource& resource,
                   IndexOwnership owned = Owned)
    : m_resource(resource), m_use_resource(true)
  {
    initIndexData(m_use_resource,
                  values, length, owned);
  }

  ///
  /// Construct list segment from arbitrary object holding
  /// indices using a deep copy of given data.
  ///
  /// The object must provide methods: begin(), end(), size().
  ///
  template <typename Container>
  TypedListSegment(const Container& container,
                   camp::resources::Resource& resource)
    : m_resource(resource), m_use_resource(true),
      m_owned(Unowned), m_data(nullptr), m_size(container.size())
  {

    if (m_size > 0) {

      camp::resources::Resource host_res{camp::resources::Host()};

      value_type* tmp = host_res.allocate<value_type>(m_size);

      auto dest = tmp;
      auto src = container.begin();
      auto const end = container.end();
      while (src != end) {
        *dest = *src;
        ++dest;
        ++src;
      }

      m_data = m_resource.allocate<value_type>(m_size);
      m_resource.memcpy(m_data, tmp, sizeof(value_type) * m_size);
      m_owned = Owned;

      host_res.deallocate(tmp);

    }
  }


/*
 * The following two ctors preserve the original list segment behavior for
 * CUDA and HIP device memory management. 
 *
 * Note that the host resource object created in the member initialization
 * list is not used. Where memory management routines are shared between
 * the old way and using camp resources are controlled by the m_use_resource
 * boolean member.
 */

  ///
  /// \brief Construct list segment from given array with specified length.
  ///
  /// By default the ctor performs deep copy of array elements.
  ///
  /// If 'Unowned' is passed as last argument, the constructed object
  /// does not own the segment data and will hold a pointer to given
  /// array's data. In this case, caller must manage object lifetimes properly.
  ///
  RAJA_DEPRECATE("In next RAJA release, TypedListSegment ctor will require a camp Resource object")
  TypedListSegment(const value_type* values,
                   Index_type length,
                   IndexOwnership owned = Owned)
    : m_resource(camp::resources::Resource{camp::resources::Host()}),
      m_use_resource(false),
      m_owned(Unowned), m_data(nullptr), m_size(0)
  {
    initIndexData(m_use_resource,
                  values, length, owned);
  }

  ///
  /// Construct list segment from arbitrary object holding
  /// indices using a deep copy of given data.
  ///
  /// The object must provide methods: begin(), end(), size().
  ///
  template <typename Container>
  RAJA_DEPRECATE("In next RAJA release, TypedListSegment ctor will require a camp Resource object")
  explicit TypedListSegment(const Container& container)
    : m_resource(camp::resources::Resource{camp::resources::Host()}),
      m_use_resource(false),
      m_owned(Unowned), m_data(nullptr), m_size(container.size())
  {
    if (m_size > 0) {
      allocate_and_copy<Has_GPU>(container);
      m_owned = Owned;
    }
  }

  ///
  /// Copy-constructor for list segment.
  ///
  TypedListSegment(const TypedListSegment& other)
    : m_resource(other.m_resource), m_use_resource(other.m_use_resource),
      m_owned(Unowned), m_data(nullptr), m_size(0)
  {
    bool from_copy_ctor = true;
    initIndexData(other.m_use_resource,
                  other.m_data, other.m_size, other.m_owned, from_copy_ctor);
  }

  ///
  /// Move-constructor for list segment.
  ///
  TypedListSegment(TypedListSegment&& rhs)
    : m_resource(rhs.m_resource), m_use_resource(rhs.m_use_resource),
      m_owned(rhs.m_owned), m_data(rhs.m_data), m_size(rhs.m_size)
  {
    // make the rhs non-owning so it's destructor won't have any side effects
    rhs.m_owned = Unowned;
  }

  ///
  /// Destroy segment including its contents
  ///
  ~TypedListSegment()
  {
    if (m_data != nullptr && m_owned == Owned) {

      if (m_use_resource) {
        m_resource.deallocate(m_data);
      } else {
        deallocate(std::integral_constant<bool, Has_GPU>());
      }

    }
  }


  ///
  /// Swap function for copy-and-swap idiom.
  ///
  RAJA_HOST_DEVICE void swap(TypedListSegment& other)
  {
    camp::safe_swap(m_resource, other.m_resource);
    camp::safe_swap(m_use_resource, other.m_use_resource);
    camp::safe_swap(m_data, other.m_data);
    camp::safe_swap(m_size, other.m_size);
    camp::safe_swap(m_owned, other.m_owned);
  }

  //! accessor to get the end iterator for a TypedListSegment
  RAJA_HOST_DEVICE iterator end() const { return m_data + m_size; }

  //! accessor to get the begin iterator for a TypedListSegment
  RAJA_HOST_DEVICE iterator begin() const { return m_data; }

  //! accessor to retrieve the total number of elements in a TypedListSegment
  RAJA_HOST_DEVICE Index_type size() const { return m_size; }

  //! get ownership of the data (Owned/Unowned)
  RAJA_HOST_DEVICE IndexOwnership getIndexOwnership() const { return m_owned; }

  //! checks a pointer and size (Span) for equality to all elements in the
  //! TypedListSegment
  RAJA_HOST_DEVICE bool indicesEqual(const value_type* container,
                                     Index_type len) const
  {
    if (container == m_data) return len == m_size;
    if (len != m_size || container == nullptr || m_data == nullptr)
      return false;
    for (Index_type i = 0; i < m_size; ++i)
      if (m_data[i] != container[i]) return false;
    return true;
  }

  ///
  /// Equality operator returns true if segments are equal; else false.
  ///
  RAJA_HOST_DEVICE bool operator==(const TypedListSegment& other) const
  {
    return (indicesEqual(other.m_data, other.m_size));
  }

  ///
  /// Inequality operator returns true if segments are not equal, else false.
  ///
  RAJA_HOST_DEVICE bool operator!=(const TypedListSegment& other) const
  {
    return (!(*this == other));
  }

private:
  //
  // Initialize segment data properly based on whether object
  // owns the index data.
  //
  void initIndexData(bool use_resource,
                     const value_type* container,
                     Index_type len,
                     IndexOwnership container_own,
                     bool from_copy_ctor = false)
  {

    // empty list segment
    if (len <= 0 || container == nullptr) {
      m_data = nullptr;
      m_size = 0;
      m_owned = Unowned;
      return;
    }

    // some non-zero size -- initialize accordingly
    m_size = len;
    m_owned = container_own;
    if (m_owned == Owned) {

      if (use_resource) {

        if ( from_copy_ctor ) {

          m_data = m_resource.allocate<value_type>(m_size);
          m_resource.memcpy(m_data, container, sizeof(value_type) * m_size); 

        } else {

          camp::resources::Resource host_res{camp::resources::Host()};

          value_type* tmp = host_res.allocate<value_type>(m_size);

          for (Index_type i = 0; i < m_size; ++i) {
            tmp[i] = container[i];
          }

          m_data = m_resource.allocate<value_type>(m_size);
          m_resource.memcpy(m_data, tmp, sizeof(value_type) * m_size);

          host_res.deallocate(tmp);

        }

      } else {
        allocate_and_copy<Has_GPU>(RAJA::make_span(container, len));
      }

      return;
    }
 
    // list segment accesses container data directly.
    // Uh-oh. Using evil const_cast....
    m_data = const_cast<value_type*>(container);
  }


  // Copy of camp resource passed to ctor
  camp::resources::Resource m_resource;

  // Boolean indicating whether camp resource is used to manage index data
  bool m_use_resource;

  // ownership flag to guide data copying/management
  IndexOwnership m_owned;

  // buffer storage for list data
  value_type* RAJA_RESTRICT m_data;

  // size of list segment
  Index_type m_size;
};

//! alias for A TypedListSegment with storage type @Index_type
using ListSegment = TypedListSegment<Index_type>;

}  // namespace RAJA

namespace std
{

/*!
 *  Specialization of std::swap for TypedListSegment
 */
template <typename T>
RAJA_INLINE void swap(RAJA::TypedListSegment<T>& a,
                      RAJA::TypedListSegment<T>& b)
{
  a.swap(b);
}
}  // namespace std

#endif  // closing endif for header file include guard
