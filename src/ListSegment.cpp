/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for list segment classes
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/index/ListSegment.hpp"

#include <iostream>
#include <string>

namespace RAJA
{

/*
*************************************************************************
*
* Public ListSegment class methods.
*
*************************************************************************
*/

////
////
ListSegment::ListSegment(const Index_type* indx,
                         Index_type len,
                         IndexOwnership indx_own)
    : BaseSegment(_ListSeg_), m_indx(0), m_len(0)
{
  initIndexData(indx, len, indx_own);
}

////
////
ListSegment::ListSegment(const ListSegment& other)
    : BaseSegment(_ListSeg_), m_indx(0), m_len(0)
{
  initIndexData(other.m_indx, other.getLength(), other.m_indx_own);
}

////
////
ListSegment& ListSegment::operator=(const ListSegment& rhs)
{
  if (&rhs != this) {
    ListSegment copy(rhs);
    this->swap(copy);
  }
  return *this;
}

////
////
ListSegment::~ListSegment()
{
  if (m_indx && m_indx_own == Owned) {
#if defined(RAJA_ENABLE_CUDA)
    cudaErrchk(cudaFree(m_indx));
#else
    delete[] m_indx;
#endif
  }
}

////
////
void ListSegment::swap(ListSegment& other)
{
  using std::swap;
  swap(m_indx, other.m_indx);
  swap(m_len, other.m_len);
  swap(m_indx_own, other.m_indx_own);
}

////
////
bool ListSegment::indicesEqual(const Index_type* indx, Index_type len) const
{
  bool equal = true;

  if (len != m_len || indx == 0 || m_indx == 0) {
    equal = false;

  } else {
    Index_type i = 0;
    while (equal && i < m_len) {
      equal = (m_indx[i] == indx[i]);
      i++;
    }
  }

  return equal;
}

////
////
void ListSegment::print(std::ostream& os) const
{
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
#endif
  os << "ListSegment : length, owns index = " << getLength()
     << (m_indx_own == Owned ? " -- Owned" : " -- Unowned") << std::endl;
  for (Index_type i = 0; i < getLength(); ++i) {
    os << "\t" << m_indx[i] << std::endl;
  }
}

/*
*************************************************************************
*
* Private initialization method.
*
*************************************************************************
*/
void ListSegment::initIndexData(const Index_type* indx,
                                Index_type len,
                                IndexOwnership indx_own)
{
  if (len <= 0 || indx == 0) {
    m_indx = 0;
    m_len = 0;
    m_indx_own = Unowned;

  } else {
    m_len = len;
    m_indx_own = indx_own;

    if (m_indx_own == Owned) {
#if defined(RAJA_ENABLE_CUDA)
      cudaErrchk(cudaMallocManaged((void**)&m_indx,
                                   m_len * sizeof(Index_type),
                                   cudaMemAttachGlobal));
      cudaErrchk(cudaMemcpy(
          m_indx, indx, m_len * sizeof(Index_type), cudaMemcpyDefault));
#else
      m_indx = new Index_type[len];
      for (Index_type i = 0; i < m_len; ++i) {
        m_indx[i] = indx[i];
      }
#endif
    } else {
      // Uh-oh. Using evil const_cast....
      m_indx = const_cast<Index_type*>(indx);
    }
  }
}

}  // closing brace for RAJA namespace
