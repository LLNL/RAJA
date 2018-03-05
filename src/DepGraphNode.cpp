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
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <string>
#include <iostream>

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

}  // closing brace for RAJA namespace
