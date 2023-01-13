/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for dependency graph node class.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <iostream>
#include <string>

#include "RAJA/internal/DepGraphNode.hpp"

namespace RAJA
{

void DepGraphNode::print(std::ostream& os) const
{
  os << "DepGraphNode : sem, reload value = " << m_semaphore_value << " , "
     << m_semaphore_reload_value << std::endl;

  os << "     num dep tasks = " << m_num_dep_tasks;
  if (m_num_dep_tasks > 0) {
    os << " ( ";
    for (int jj = 0; jj < m_num_dep_tasks; ++jj) {
      os << m_dep_task[jj] << "  ";
    }
    os << " )";
  }
  os << std::endl;
}

}  // namespace RAJA
