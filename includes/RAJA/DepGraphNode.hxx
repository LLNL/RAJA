/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining simple class to manage scheduling
 *          of nodes in a task dependency graph.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_DepGraphNode_HXX
#define RAJA_DepGraphNode_HXX

#include "int_datatypes.hxx"

#include <cstdio>
#include <cstdlib>

#include <iosfwd>

namespace RAJA {

/*!
 ******************************************************************************
 *
 * \brief  Class defining a simple semephore-based data structure for 
 *         managing a node in a dependency graph.
 *
 ******************************************************************************
 */
class DepGraphNode
{
public:

   ///
   /// Constants for total number of allowable dependent tasks 
   /// and alignment of semaphore value (should match cache 
   /// coherence page size?).
   ///
   /// These values may need to be set differently for different
   /// algorithms and platforms.  We haven't figured this out yet!
   ///
   static const int _MaxDepTasks_          = 8;
   static const int _SemaphoreValueAlign_  = 256;

   ///
   /// Default ctor initializes node to default state.
   ///
   DepGraphNode()
      : m_num_dep_tasks(0),
        m_semaphore_reload_value(0),
        m_semaphore_value(0) 
   { 
      posix_memalign((void **)(&m_semaphore_value), 
                     _SemaphoreValueAlign_, sizeof(int)) ;
      *m_semaphore_value = 0 ;       
   }

   ///
   /// Dependency graph node dtor.
   ///
   ~DepGraphNode() { if (m_semaphore_value) free(m_semaphore_value); }

   ///
   /// Get/set semaphore value; i.e., the current number of (unsatisfied)
   /// dependencies that must be satisfied before this task can execute.
   ///
   int& semaphoreValue() {
      return *m_semaphore_value ;
   }

   ///
   /// Get/set semaphore "reload" value; i.e., the total number of external
   /// task dependencies that must be satisfied before this task can execute.
   ///
   int& semaphoreReloadValue() {
      return m_semaphore_reload_value ;
   }

   ///
   /// Get/set the number of "forward-dependencies" for this task; i.e., the
   /// number of external tasks that cannot execute until this task completes. 
   ///
   int& numDepTasks() {
      return m_num_dep_tasks ;
   }

   ///
   /// Get/set the forward dependency task number associated with the given 
   /// index for this task. This is used to notify the appropriate external 
   /// dependencies when this task completes.
   ///
   int& depTaskNum(int tidx) {
      return m_dep_task[tidx] ;
   }


   ///
   /// Print task graph object node data to given output stream.
   ///
   void print(std::ostream& os) const;

private:

   int  m_dep_task[_MaxDepTasks_] ;
   int  m_num_dep_tasks ;
   int  m_semaphore_reload_value ;
   int* m_semaphore_value ;

}; 


}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
