#ifndef RAJA_pattern_nested_ShmemWindow_HPP
#define RAJA_pattern_nested_ShmemWindow_HPP


#include "RAJA/config.hpp"
#include "RAJA/util/StaticLayout.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{

namespace nested
{


/*!
 * A nested::forall statement that sets the shared memory window.
 *
 *
 */

template <typename... EnclosedStmts>
struct SetShmemWindow
    : public internal::Statement<camp::nil, EnclosedStmts...> {
};


namespace internal
{




template <typename... EnclosedStmts>
struct StatementExecutor<SetShmemWindow<EnclosedStmts...>> {


  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {

    // Call setWindow on all of our shmem objects
    shmem_set_windows(data.param_tuple, data.get_begin_index_tuple());

    // Invoke the enclosed statements
    execute_statement_list<camp::list<EnclosedStmts...>>(
        std::forward<Data>(data));
  }

};





}  // namespace internal
}  // end namespace nested
}  // end namespace RAJA


#endif /* RAJA_pattern_nested_HPP */
