#ifndef RAJA_POLICYBASE_HXX
#define RAJA_POLICYBASE_HXX

#include "RAJA/int_datatypes.hxx"

namespace RAJA {

struct PolicyBase {
    template<typename IndexT = Index_type,
             typename Func,
             typename std::enable_if<!std::is_base_of<
                 std::random_access_iterator_tag,
                 typename std::iterator_traits<IndexT>::iterator_category>::value>::type * = nullptr>
    void operator()(IndexT begin, IndexT end, Func &&f) const {
        for ( auto ii = begin ; ii < end ; ++ii ) {
            loop_body( ii );
        }
    }

    template<typename Iterator,
             typename Func>
    void operator()(Iterator &&begin, Iterator &&end, Func &&loop_body) const {
        for ( auto &ii = begin ; ii < end ; ++ii ) {
            loop_body( *ii );
        }
    }
};

}

#endif /* RAJA_POLICYBASE_HXX */
