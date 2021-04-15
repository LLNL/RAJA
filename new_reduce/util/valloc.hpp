#ifndef VALLOC_HPP
#define VALLOC_HPP

template<typename T, bool min = true>
class ValLoc {
  T val = min ? RAJA::operators::limits<T>::max() : RAJA::operators::limits<T>::min();
  RAJA::Index_type loc;
public:

  ValLoc() : loc(-1) {}
  ValLoc(T v) : val(v), loc(-1) {}
  ValLoc(T v, RAJA::Index_type l) : val(v), loc(l) {}

  ValLoc constexpr operator () (T v, RAJA::Index_type l) {
    if (min) 
      {if (v < val) { val = v; loc = l; } } 
    else
      {if (v > val) { val = v; loc = l; } }
    return *this;
  }

  bool constexpr operator < (const ValLoc& rhs) const { return val < rhs.val; }
  bool constexpr operator <=(const ValLoc& rhs) const { return val < rhs.val; }
  bool constexpr operator > (const ValLoc& rhs) const { return val > rhs.val; }
  bool constexpr operator >=(const ValLoc& rhs) const { return val > rhs.val; }

  T getVal() {return val;}
  RAJA::Index_type getLoc() {return loc;}
};

template<typename T>
ValLoc<T>& make_valloc(T v, RAJA::Index_type l) { return ValLoc<T>(v, l); }

template<typename T>
using ValLocMin = ValLoc<T, true>;

template<typename T>
using ValLocMax = ValLoc<T, false>;


namespace RAJA
{

namespace operators
{

template <typename T>
struct limits<ValLocMin<T>> {
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr ValLocMin<T> min()
  {
    return ValLocMin<T>(RAJA::operators::limits<T>::min());
  }
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr ValLocMin<T> max()
  {
    return ValLocMin<T>(RAJA::operators::limits<T>::max());
  }
};
template <typename T>
struct limits<ValLocMax<T>> {
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr ValLocMax<T> min()
  {
    return ValLocMax<T>(RAJA::operators::limits<T>::min());
  }
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr ValLocMax<T> max()
  {
    return ValLocMax<T>(RAJA::operators::limits<T>::max());
  }
};

} //  namespace operators

} //  namespace RAJA

#endif
