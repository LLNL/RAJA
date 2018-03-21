#ifndef CAMP_CONCEPTS_HPP
#define CAMP_CONCEPTS_HPP

#include <iterator>
#include <type_traits>

#include "camp/helpers.hpp"
#include "camp/number.hpp"

namespace camp
{

namespace concepts
{

  namespace metalib
  {

    template <typename T, typename U>
    struct is_same_s : false_type {
    };

    template <typename T>
    struct is_same_s<T, T> : true_type {
    };

#if defined(CAMP_COMPILER_MSVC)
    template <typename... Ts>
    using is_same = typename is_same_s<Ts...>::type;
#else
    template <typename T, typename U>
    using is_same = typename is_same_s<T, U>::type;
#endif


    /// negation metafunction of a value type
    template <typename T>
    struct negate_t : num<!T::value> {
    };

    /// all_of metafunction of a value type list -- all must be "true"
    template <bool... Bs>
    struct all_of : metalib::is_same<list<t, num<Bs>...>, list<num<Bs>..., t>> {
    };

    /// none_of metafunction of a value type list -- all must be "false"
    template <bool... Bs>
    struct none_of
        : metalib::is_same<idx_seq<false, Bs...>, idx_seq<Bs..., false>> {
    };

    /// any_of metafunction of a value type list -- at least one must be "true""
    template <bool... Bs>
    struct any_of : negate_t<none_of<Bs...>> {
    };

    /// all_of metafunction of a bool list -- all must be "true"
    template <typename... Bs>
    struct all_of_t : all_of<Bs::value...> {
    };

    /// none_of metafunction of a bool list -- all must be "false"
    template <typename... Bs>
    struct none_of_t : none_of<Bs::value...> {
    };

    /// any_of metafunction of a bool list -- at least one must be "true""
    template <typename... Bs>
    struct any_of_t : any_of<Bs::value...> {
    };

  }  // end namespace metalib

}  // end namespace concepts
}  // end namespace camp

template <typename... T>
camp::true_type ___valid_expr___(T &&...) noexcept;
#define DefineConcept(...) decltype(___valid_expr___(__VA_ARGS__))

#define DefineTypeTraitFromConcept(TTName, ConceptName)             \
  template <typename... Args>                                       \
  struct TTName : camp::concepts::requires_<ConceptName, Args...> { \
  }
namespace camp
{
namespace concepts
{

  namespace detail
  {

    template <class...>
    struct TL {
    };

    template <class...>
    struct voider {
      using type = void;
    };

    template <class Default,
              class /* always void*/,
              template <class...> class Concept,
              class TArgs>
    struct detector {
      using value_t = false_type;
      using type = Default;
    };

    template <class Default, template <class...> class Concept, class... Args>
    struct detector<Default,
                    typename voider<Concept<Args...>>::type,
                    Concept,
                    TL<Args...>> {
      using value_t = true_type;
      using type = Concept<Args...>;
    };

    template <template <class...> class Concept, class TArgs>
    using is_detected = detector<void, void, Concept, TArgs>;

    template <template <class...> class Concept, class TArgs>
    using detected = typename is_detected<Concept, TArgs>::value_t;


    template <typename Ret, typename T>
    Ret returns(T const &) noexcept;

  }  // end namespace detail

  template <typename T>
  using negate = metalib::negate_t<T>;

  /// metafunction for use within decltype expression to validate return type is
  /// convertible to given type
  template <typename T, typename U>
  constexpr auto convertible_to(U &&u) noexcept
      -> decltype(detail::returns<camp::true_type>(static_cast<T>((U &&) u)));

  /// metafunction for use within decltype expression to validate type of
  /// expression
  template <typename T, typename U>
  constexpr auto has_type(U &&) noexcept -> metalib::is_same<T, U>;

  template <typename BoolLike>
  constexpr auto is(BoolLike) noexcept
      -> camp::if_<BoolLike, camp::true_type, camp::false_type>;

  template <typename BoolLike>
  constexpr auto is_not(BoolLike) noexcept
      -> camp::if_c<!BoolLike::value, camp::true_type, camp::false_type>;

  /// metaprogramming concept for SFINAE checking of aggregating concepts
  template <typename... Args>
  struct all_of : metalib::all_of_t<Args...> {
  };

  /// metaprogramming concept for SFINAE checking of aggregating concepts
  template <typename... Args>
  struct none_of : metalib::none_of_t<Args...> {
  };

  /// metaprogramming concept for SFINAE checking of aggregating concepts
  template <typename... Args>
  struct any_of : metalib::any_of_t<Args...> {
  };

  /// SFINAE multiple type traits
  template <typename... Args>
  using enable_if = typename std::enable_if<all_of<Args...>::value, void>::type;

  /// SFINAE concept checking
  template <template <class...> class Op, class... Args>
  struct requires_ : detail::detected<Op, detail::TL<Args...>> {
  };

  template <typename T>
  struct Swappable : DefineConcept(swap(val<T>(), val<T>())) {
  };

  template <typename T>
  struct LessThanComparable
      : DefineConcept(convertible_to<bool>(val<T>() < val<T>())) {
  };

  template <typename T>
  struct GreaterThanComparable
      : DefineConcept(convertible_to<bool>(val<T>() > val<T>())) {
  };

  template <typename T>
  struct LessEqualComparable
      : DefineConcept(convertible_to<bool>(val<T>() <= val<T>())) {
  };

  template <typename T>
  struct GreaterEqualComparable
      : DefineConcept(convertible_to<bool>(val<T>() >= val<T>())) {
  };

  template <typename T>
  struct EqualityComparable
      : DefineConcept(convertible_to<bool>(val<T>() == val<T>())) {
  };

  template <typename T, typename U>
  struct ComparableTo
      : DefineConcept(convertible_to<bool>(val<U>() < val<T>()),
                      convertible_to<bool>(val<T>() < val<U>()),
                      convertible_to<bool>(val<U>() <= val<T>()),
                      convertible_to<bool>(val<T>() <= val<U>()),
                      convertible_to<bool>(val<U>() > val<T>()),
                      convertible_to<bool>(val<T>() > val<U>()),
                      convertible_to<bool>(val<U>() >= val<T>()),
                      convertible_to<bool>(val<T>() >= val<U>()),
                      convertible_to<bool>(val<U>() == val<T>()),
                      convertible_to<bool>(val<T>() == val<U>()),
                      convertible_to<bool>(val<U>() != val<T>()),
                      convertible_to<bool>(val<T>() != val<U>())) {
  };

  template <typename T>
  struct Comparable : ComparableTo<T, T> {
  };

  template <typename T>
  struct Arithmetic : DefineConcept(is(std::is_arithmetic<T>())) {
  };

  template <typename T>
  struct FloatingPoint : DefineConcept(is(std::is_floating_point<T>())) {
  };

  template <typename T>
  struct Integral : DefineConcept(is(std::is_integral<T>())) {
  };

  template <typename T>
  struct Signed : DefineConcept(Integral<T>(), is(std::is_signed<T>())) {
  };

  template <typename T>
  struct Unsigned : DefineConcept(Integral<T>(), is(std::is_unsigned<T>())) {
  };

  template <typename T>
  struct Iterator
      : DefineConcept(is_not(Integral<T>()),  // hacky NVCC 8 workaround
                      *(val<T>()),
                      has_type<T &>(++val<T &>())) {
  };

  template <typename T>
  struct ForwardIterator
      : DefineConcept(Iterator<T>(), val<T &>()++, *val<T &>()++) {
  };

  template <typename T>
  struct BidirectionalIterator
      : DefineConcept(ForwardIterator<T>(),
                      has_type<T &>(--val<T &>()),
                      convertible_to<T const &>(val<T &>()--),
                      *val<T &>()--) {
  };

  template <typename T>
  struct RandomAccessIterator
      : DefineConcept(BidirectionalIterator<T>(),
                      Comparable<T>(),
                      has_type<T &>(val<T &>() += val<diff_from<T>>()),
                      has_type<T>(val<T>() + val<diff_from<T>>()),
                      has_type<T>(val<diff_from<T>>() + val<T>()),
                      has_type<T &>(val<T &>() -= val<diff_from<T>>()),
                      has_type<T>(val<T>() - val<diff_from<T>>()),
                      val<T>()[val<diff_from<T>>()]) {
  };

  template <typename T>
  struct HasBeginEnd : DefineConcept(std::begin(val<T>()), std::end(val<T>())) {
  };

  template <typename T>
  struct Range : DefineConcept(HasBeginEnd<T>(), Iterator<iterator_from<T>>()) {
  };

  template <typename T>
  struct ForwardRange
      : DefineConcept(HasBeginEnd<T>(), ForwardIterator<iterator_from<T>>()) {
  };

  template <typename T>
  struct BidirectionalRange
      : DefineConcept(HasBeginEnd<T>(),
                      BidirectionalIterator<iterator_from<T>>()) {
  };

  template <typename T>
  struct RandomAccessRange
      : DefineConcept(HasBeginEnd<T>(),
                      RandomAccessIterator<iterator_from<T>>()) {
  };

}  // end namespace concepts

namespace type_traits
{
  DefineTypeTraitFromConcept(is_iterator, camp::concepts::Iterator);
  DefineTypeTraitFromConcept(is_forward_iterator,
                             camp::concepts::ForwardIterator);
  DefineTypeTraitFromConcept(is_bidirectional_iterator,
                             camp::concepts::BidirectionalIterator);
  DefineTypeTraitFromConcept(is_random_access_iterator,
                             camp::concepts::RandomAccessIterator);

  DefineTypeTraitFromConcept(is_range, camp::concepts::Range);
  DefineTypeTraitFromConcept(is_forward_range, camp::concepts::ForwardRange);
  DefineTypeTraitFromConcept(is_bidirectional_range,
                             camp::concepts::BidirectionalRange);
  DefineTypeTraitFromConcept(is_random_access_range,
                             camp::concepts::RandomAccessRange);

  DefineTypeTraitFromConcept(is_comparable, camp::concepts::Comparable);
  DefineTypeTraitFromConcept(is_comparable_to, camp::concepts::ComparableTo);

  DefineTypeTraitFromConcept(is_arithmetic, camp::concepts::Arithmetic);
  DefineTypeTraitFromConcept(is_floating_point, camp::concepts::FloatingPoint);
  DefineTypeTraitFromConcept(is_integral, camp::concepts::Integral);
  DefineTypeTraitFromConcept(is_signed, camp::concepts::Signed);
  DefineTypeTraitFromConcept(is_unsigned, camp::concepts::Unsigned);

  template <typename T>
  using IterableValue = decltype(*std::begin(camp::val<T>()));

  template <typename T>
  using IteratorValue = decltype(*camp::val<T>());

  namespace detail
  {

    template <typename, template <typename...> class, typename...>
    struct IsSpecialized : camp::false_type {
    };

    template <template <typename...> class Template, typename... T>
    struct IsSpecialized<typename concepts::detail::voider<decltype(
                             camp::val<Template<T...>>())>::type,
                         Template,
                         T...> : camp::true_type {
    };

    template <template <class...> class,
              template <class...> class,
              bool,
              class...>
    struct SpecializationOf : camp::false_type {
    };

    template <template <class...> class Expected,
              template <class...> class Actual,
              class... Args>
    struct SpecializationOf<Expected, Actual, true, Args...>
        : camp::concepts::metalib::is_same<Expected<Args...>, Actual<Args...>> {
    };

  }  // end namespace detail


  template <template <class...> class Outer, class... Args>
  using IsSpecialized = detail::IsSpecialized<void, Outer, Args...>;

  template <template <class...> class, typename T>
  struct SpecializationOf : camp::false_type {
  };

  template <template <class...> class Expected,
            template <class...> class Actual,
            class... Args>
  struct SpecializationOf<Expected, Actual<Args...>>
      : detail::SpecializationOf<Expected,
                                 Actual,
                                 IsSpecialized<Expected, Args...>::value,
                                 Args...> {
  };

}  // end namespace type_traits
}  // namespace camp

#endif /* CAMP_CONCEPTS_HPP */
