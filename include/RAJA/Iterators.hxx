#ifndef RAJA_ITERATORS_HXX
#define RAJA_ITERATORS_HXX

#include "RAJA/int_datatypes.hxx"
#include "RAJA/internal/defines.hxx"

#include <type_traits>
#include <iterator>
#include <utility>

namespace RAJA {
    namespace Iterators {


// Helpers


template<typename Container>
using IteratorCategoryOf = typename std::iterator_traits<typename std::remove_reference<Container>::type::iterator>::iterator_category;

template<typename Container>
using OffersRAI =
    std::is_base_of<
        std::random_access_iterator_tag,
        IteratorCategoryOf<Container>>;

// Containers

template<typename Type,
         typename DifferenceType = std::ptrdiff_t,
         typename PointerType = Type *>
class base_iterator : public std::iterator<std::random_access_iterator_tag,
                                                Type,
                                                DifferenceType>
{
public:
    using difference_type = typename std::iterator<std::random_access_iterator_tag, Type>::difference_type;

    RAJA_HOST_DEVICE constexpr base_iterator() : val(0) {}
    RAJA_HOST_DEVICE constexpr base_iterator(Type rhs) : val(rhs) {}
    RAJA_HOST_DEVICE constexpr base_iterator(const base_iterator &rhs) : val(rhs.val) {}

    RAJA_HOST_DEVICE inline bool operator==(const base_iterator& rhs) const {return val == rhs.val;}
    RAJA_HOST_DEVICE inline bool operator!=(const base_iterator& rhs) const {return val != rhs.val;}
    RAJA_HOST_DEVICE inline bool operator>(const base_iterator& rhs) const {return val > rhs.val;}
    RAJA_HOST_DEVICE inline bool operator<(const base_iterator& rhs) const {return val < rhs.val;}
    RAJA_HOST_DEVICE inline bool operator>=(const base_iterator& rhs) const {return val >= rhs.val;}
    RAJA_HOST_DEVICE inline bool operator<=(const base_iterator& rhs) const {return val <= rhs.val;}
protected:
    Type val;
};

template<typename Type = Index_type,
         typename DifferenceType = Index_type,
         typename PointerType = Type *>
class numeric_iterator : public base_iterator< Type,
                                               DifferenceType>
{
public:
    using difference_type = typename std::iterator<std::random_access_iterator_tag, Type>::difference_type;
    using base = base_iterator<Type, DifferenceType>;

    RAJA_HOST_DEVICE constexpr numeric_iterator() : base(0) {}
    RAJA_HOST_DEVICE constexpr numeric_iterator(const Type& rhs) : base(rhs) {}
    RAJA_HOST_DEVICE constexpr numeric_iterator(const numeric_iterator& rhs) : base(rhs.val) {}

    RAJA_HOST_DEVICE inline numeric_iterator& operator++() {++base::val; return *this;}
    RAJA_HOST_DEVICE inline numeric_iterator& operator--() {--base::val; return *this;}
    RAJA_HOST_DEVICE inline numeric_iterator operator++(int) {numeric_iterator tmp(*this); ++base::val; return tmp;}
    RAJA_HOST_DEVICE inline numeric_iterator operator--(int) {numeric_iterator tmp(*this); --base::val; return tmp;}

    RAJA_HOST_DEVICE inline numeric_iterator& operator+=(const difference_type& rhs) {base::val+=rhs; return *this;}
    RAJA_HOST_DEVICE inline numeric_iterator& operator-=(const difference_type& rhs) {base::val-=rhs; return *this;}
    RAJA_HOST_DEVICE inline numeric_iterator& operator+=(const numeric_iterator& rhs) {base::val+=rhs.val; return *this;}
    RAJA_HOST_DEVICE inline numeric_iterator& operator-=(const numeric_iterator& rhs) {base::val-=rhs.val; return *this;}

    RAJA_HOST_DEVICE inline difference_type operator+(const numeric_iterator& rhs) const {return static_cast<difference_type>(base::val)+static_cast<difference_type>(rhs.val);}
    RAJA_HOST_DEVICE inline difference_type operator-(const numeric_iterator& rhs) const {return static_cast<difference_type>(base::val)-static_cast<difference_type>(rhs.val);}
    RAJA_HOST_DEVICE inline numeric_iterator operator+(const difference_type& rhs) const {return numeric_iterator(base::val+rhs);}
    RAJA_HOST_DEVICE inline numeric_iterator operator-(const difference_type& rhs) const {return numeric_iterator(base::val-rhs);}
    RAJA_HOST_DEVICE friend constexpr numeric_iterator operator+(difference_type lhs, const numeric_iterator& rhs) {return numeric_iterator(lhs+rhs.val);}
    RAJA_HOST_DEVICE friend constexpr numeric_iterator operator-(difference_type lhs, const numeric_iterator& rhs) {return numeric_iterator(lhs-rhs.val);}

    RAJA_HOST_DEVICE inline Type operator*() const {return base::val;}
    RAJA_HOST_DEVICE inline Type operator->() const {return base::val;}
    RAJA_HOST_DEVICE constexpr Type operator[](difference_type rhs) const {return base::val + rhs;}

};

template<typename Type = Index_type,
         typename DifferenceType = Index_type,
         typename PointerType = Type *>
class strided_numeric_iterator : public base_iterator< Type,
                                               DifferenceType>
{
public:
    using difference_type = typename std::iterator<std::random_access_iterator_tag, Type>::difference_type;
    using base = base_iterator<Type, DifferenceType>;

    RAJA_HOST_DEVICE constexpr strided_numeric_iterator() : base(0), stride(1) {}
    RAJA_HOST_DEVICE constexpr strided_numeric_iterator(const Type& rhs, DifferenceType stride = 1) : base(rhs), stride(stride) {}
    RAJA_HOST_DEVICE constexpr strided_numeric_iterator(const strided_numeric_iterator& rhs) : base(rhs.val), stride(rhs.stride) {}

    RAJA_HOST_DEVICE inline strided_numeric_iterator& operator++() {base::val += stride; return *this;}
    RAJA_HOST_DEVICE inline strided_numeric_iterator& operator--() {base::val -= stride; return *this;}

    RAJA_HOST_DEVICE inline strided_numeric_iterator& operator+=(const difference_type& rhs) {base::val+=rhs * stride; return *this;}
    RAJA_HOST_DEVICE inline strided_numeric_iterator& operator-=(const difference_type& rhs) {base::val-=rhs * stride; return *this;}

    RAJA_HOST_DEVICE inline difference_type operator+(const strided_numeric_iterator& rhs) const {
        return static_cast<difference_type>(base::val)
            + (static_cast<difference_type>(rhs.val * stride));}
    RAJA_HOST_DEVICE inline difference_type operator-(const strided_numeric_iterator& rhs) const {
        auto diff = static_cast<difference_type>(base::val)
            - (static_cast<difference_type>(rhs.val));
        if (diff < stride)
            return 0;
        if (diff % stride) // check for off-stride endpoint
            return diff/stride + 1;
        return diff/stride;
    }
    RAJA_HOST_DEVICE inline strided_numeric_iterator operator+(const difference_type& rhs) const {return strided_numeric_iterator(base::val+rhs * stride);}
    RAJA_HOST_DEVICE inline strided_numeric_iterator operator-(const difference_type& rhs) const {return strided_numeric_iterator(base::val-rhs * stride);}

    // Specialized comparison to allow normal iteration to work on off-stride
    // multiples by adjusting rhs to the nearest *higher* multiple of stride
    RAJA_HOST_DEVICE inline bool operator!=(const strided_numeric_iterator& rhs) const {
        if (base::val == rhs.val) return false;
        auto rem = rhs.val % stride;
        return base::val != rhs.val + rem;
    }

    RAJA_HOST_DEVICE inline Type operator*() const {return base::val;}
    RAJA_HOST_DEVICE inline Type operator->() const {return base::val;}
    RAJA_HOST_DEVICE inline Type operator[](difference_type rhs) const {return base::val + rhs * stride;}

private:
    DifferenceType stride;

};

// TODO: this should really be a generic Zip, then using Enumerator =
// Zip<numeric_iterator, Iterator>
template<typename Iterator>
class Enumerater : public numeric_iterator<> {
public:
    template<typename First,
             typename Second>
    class InnerPair {
        public:
        InnerPair() = delete;
        RAJA_HOST_DEVICE constexpr InnerPair(First && f, Second && s) : first(f), second(s) {};
        First first;
        Second second;
    };
    using base = numeric_iterator<>;

    using pair = InnerPair<std::ptrdiff_t,
                           Iterator>;
    using value_type = pair;
    using pointer_type = pair*;
    using reference = pair&;

    Enumerater() = delete;
    RAJA_HOST_DEVICE constexpr Enumerater(const Iterator& rhs,
                         std::ptrdiff_t val = 0,
                         std::ptrdiff_t offset = 0)
        : base(val), offset(offset), wrapped(rhs) {}
    RAJA_HOST_DEVICE constexpr Enumerater(const Enumerater &rhs)
        : base(rhs.val), offset(rhs.offset), wrapped(rhs.wrapped) {}

    RAJA_HOST_DEVICE inline pair operator*() const {return pair(offset+val, wrapped+val);}
    RAJA_HOST_DEVICE constexpr pair operator[](typename base::difference_type rhs) const {
        return pair(val+offset+rhs, wrapped+val+rhs);
    }

private:
    std::ptrdiff_t offset;
    Iterator wrapped;
};

}
}

#endif /* RAJA_ITERATORS_HXX */
