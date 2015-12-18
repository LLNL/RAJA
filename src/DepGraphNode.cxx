/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for dependency graph node class.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#include "RAJA/DepGraphNode.hxx"

#include <iostream>

namespace RAJA {


void DepGraphNode::print(std::ostream& os) const
{
   os << "DepGraphNode : sem, reload value = " 
      << *m_semaphore_value << " , " << m_semaphore_reload_value << std::endl;

   os << "     num dep tasks = " << m_num_dep_tasks;
   if ( m_num_dep_tasks > 0 ) {
      os << " ( ";
      for (int jj=0; jj<m_num_dep_tasks; ++jj) {
         os << m_dep_task[jj] << "  "; 
      }
      os << " )";
   } 
   os << std::endl;
}


}  // closing brace for RAJA namespace
