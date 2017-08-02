# Concepts

### New namespaces:
* RAJA::concepts -- describes "Concepts" that RAJA provides
* RAJA::type_traits -- describes type traits of Concepts that can be used in SFINAE/static_assert contexts

## Concepts:
* Iterators: `Iterator`, `ForwardIterator`, `BidirectionalIterator`, `RandomAccessIterator`
* Ranges: `Range`, `ForwardRange`, `BidirectionalRange`, `RandomAccessRange`
* Types: `Arithmetic`, `Integral`, `Signed`, `Unsigned`, `FloatingPoint`
* Comparable: `LessThanComparable`, `GreaterThanComparable`, `LessEqualComparable`, `GreaterEqualComparable`, `EqualityComparable`, `ComparableTo<T,U>`, `Comparable`

## RAJA-Specific Concepts:
* `ExecutionPolicy` - checks to make sure a type models an ExecutionPolicy
* `BinaryFunction<Function, Return, Arg1, Arg2>` - checks to see that invoking `Function` with two arguments is valid and has a return type convertible to `Return`
* `UnaryFunction<Function, Return, Arg>` - checks to see that invoking `Function` with one argument is valid and has a return type convertible to `Return`

## Type Traits:
* Iterators: `is_iterator`, `is_forward_iterator`, `is_bidirectional_iterator`, `is_random_access_iterator`
* Ranges: `is_range`, `is_forward_range`, `is_bidirectional_range`, `is_random_access_range`
* Types: `is_arithmetic`, `is_integral`, `is_signed`, `is_unsigned`, `is_floating_point`
* Comparison: `is_comparable`, `is_comparable_to<T,U>`

## RAJA-Specific Type Traits:
* Execution Policies: `is_exection_policy`, `is_sequential_policy`, `is_simd_policy`, `is_openmp_policy`, `is_target_openmp_policy`, `is_cuda_policy`, `is_indexset_policy`
* Functions: `is_binary_function`, `is_unary_function`
* IndexSet: `is_index_set`

## Concept Components:

All building blocks live within the RAJA::concepts namespace. Defined concepts will be in UpperCamelCase while all building blocks adhere to snake_case.

### Naming convention:
* T, U, etc -- types
* B -- a boolean-like type (has `constexpr` value member of type `bool`)
* Ts..., Bs... -- a type or boolean-type variadic list
* bool-expr --expression that evaluates to true or false
* expr -- an expression

### Building Blocks for use within Concepts
* `val<T>()` -- an alias  #to `std::declval<T>()` -- types in a concept shall never be instantiated with a constructor (e.g. `T()`). `val<T>()` **must** be used
* `cval<T>()` -- alias for `val<T const>()`
* `has_type<T>(expr)` -- expression evaluates to return type of T
* `convertible_to<T>(expr)` -- expression evaluation return type is convertible to T
* `is(B)` -- results in a valid expression IFF `B::value` evaluates to true
* `is_not(B)` -- results in a valid expression IFF `B::value` evaluates to false

### Logical Reductions for use within type traits / SFINAE constructs
* `all_of<Bs...>` -- returns B where value is true IFF all `Bs::value ...` are true
* `none_of<Bs...>` -- returns B where value is true IFF all `Bs::value ...` are false
* `any_of<Bs...>` -- returns B where value is true IFF any `Bs::value ...` are true
* `negate<B>` -- returns B where value is `!B::value`
* `bool_<bool>` -- creates a BoolLike type

### Using Concepts at API/User-Level Constructs
* `enable_if<Bs...>` -- type to use to conditional enable/disable visibility. All `Bs::value ...` must evaluate to true
* `requires_<Concept, Ts...>` -- SFINAE wrapper around a concept check; usually to define Type Traits from Concepts

### Macros:

#### `DefineConcept(...)`

creates a new type expression (decltype) that can contain arbitrary code that must be valid in order for a type to conform to some defined Concept

#### `DefineTypeTraitFromConcept(type_trait_name, ConceptName)`

creates a new type called `type_trait_name` which derives from `requires_<ConceptName, ... >`. Useful for not rewriting boiler-plate code for type trait definitions

## Defining Concepts:

## Concept Definition Requirements:
* shall be defined as a new type (**not** an alias)
* shall adhere to CamelCase naming conventions
* shall exist under the `RAJA::concepts` namespace for API-provided concepts, otherwise shall exist under some `concepts` namespace which may be nested
* shall only accept `typename`s as template arguments (`template <class...> class` is not permitted)
* shall not specify any default template arguments -- prefer creating a new concept modeling the specialization of an existing Concept.
* shall not contain any type aliases, functions, or [static] [constexpr] members (`sizeof(Concept)` must always evaluate to 0)


## Examples

### Basic Example

```cpp
template <typename T>
struct Iterator : DefineConcept(*(val<T>()), has_type<T &>(++val<T &>()) {}
```

### Requiring Concepts modeling another Concept

```cpp
template <typename T>
struct ForwardIterator : DefineConcept(Iterator<T>(), val<T &>()++, *val<T &>()++) {}
```

Other usages can be found under:
* `include/RAJA/util/concepts.hpp`
* `include/RAJA/policy/PolicyBase.hpp`
* `include/RAJA/util/Operators.hpp`
