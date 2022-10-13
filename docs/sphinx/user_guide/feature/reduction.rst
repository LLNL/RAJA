.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and other RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _feat-reductions-label:

====================
Reduction Operations
====================

RAJA does not provide separate loop execution methods for loops containing
reduction operations like some other C++ loop programming abstraction models.
Instead, RAJA provides reduction types that allow users to perform reduction 
operations in kernels launched using ``RAJA::forall``, ``RAJA::kernel``,
and ``RAJA::launch`` methods in a portable, thread-safe manner. Users may 
use as many reduction objects in a loop kernel as they need. Available RAJA 
reduction types are described in this section.

.. note:: All RAJA reduction types are located in the namespace ``RAJA``.

Also

.. note:: * Each RAJA reduction type is templated on a **reduction policy** 
            and a **reduction value type** for the reduction variable. The
            **reduction policy type must be compatible with the execution
            policy used by the kernel in which it is used.** For example, in 
            a CUDA kernel, a CUDA reduction policy must be used. 
          * Each RAJA reduction type accepts an **initial reduction value or 
            values** at construction (see below).
          * Each RAJA reduction type has a 'get' method to access reduced
            values after kernel execution completes.

Please see the following tutorial sections for detailed examples that use
RAJA reductions:

 * :ref:`tut-reduction-label`.


----------------
Reduction Types
----------------

RAJA supports five common reduction types:

* ``ReduceSum< reduce_policy, data_type >`` - Sum of values.

* ``ReduceMin< reduce_policy, data_type >`` - Min value.

* ``ReduceMax< reduce_policy, data_type >`` - Max value.

* ``ReduceMinLoc< reduce_policy, data_type >`` - Min value and a loop index where the minimum was found.

* ``ReduceMaxLoc< reduce_policy, data_type >`` - Max value and a loop index where the maximum was found.

and two less common bitwise reduction types:

* ``ReduceBitAnd< reduce_policy, data_type >`` - Bitwise 'and' of values (i.e., ``a & b``).

* ``ReduceBitOr< reduce_policy, data_type >`` - Bitwise 'or' of values (i.e., ``a | b``).

.. note:: * When ``RAJA::ReduceMinLoc`` and ``RAJA::ReduceMaxLoc`` are used 
            in a sequential execution context, the loop index of the 
            min/max is the first index where the min/max occurs.
          * When these reductions are used in a parallel execution context, 
            the loop index computed for the reduction value may be any index 
            where the min or max occurs. 

.. note:: ``RAJA::ReduceBitAnd`` and ``RAJA::ReduceBitOr`` reduction types are designed to work on integral data types because **in C++, at the language level, there is no such thing as a bitwise operator on floating-point numbers.**

-------------------
Reduction Examples
-------------------

Next, we provide a few examples to illustrate basic usage of RAJA reduction
types.

Here is a simple RAJA reduction example that shows how to use a sum reduction 
type and a min-loc reduction type::

  const int N = 1000;

  //
  // Initialize array of length N with all ones. Then, set some other
  // values in the array to make the example mildly interesting...
  //
  int vec[N] = {1};
  vec[100] = -10; vec[500] = -10;

  // Create a sum reduction object with initial value of zero
  RAJA::ReduceSum< RAJA::omp_reduce, int > vsum(0);

  // Create a min-loc reduction object with initial min value of 100
  // and initial location index value of -1
  RAJA::ReduceMinLoc< RAJA::omp_reduce, int > vminloc(100, -1);

  // Run a kernel using the reduction objects
  RAJA::forall<RAJA::omp_parallel_for_exec>( RAJA::RangeSegment(0, N),
    [=](RAJA::Index_type i) {

    vsum += vec[i];
    vminloc.minloc( vec[i], i );

  });

  // After kernel is run, extract the reduced values
  int my_vsum = static_cast<int>(vsum.get());

  int my_vmin = static_cast<int>(vminloc.get());
  int my_vminloc = static_cast<int>(vminloc.getLoc());

The results of these operations will yield the following values:

 * my_vsum == 978 (= 998 - 10 - 10)
 * my_vmin == -10
 * my_vminloc == 100 or 500 

Note that the location index for the minimum array value can be one of two
values depending on the order of the reduction finalization since the loop
is run in parallel. Also, note that the reduction objects are created using
a ``RAJA::omp_reduce`` reduction policy, which is compatible with the 
OpenMP execution policy used in the kernel.

Here is an example of a bitwise or reduction::

  const int N = 100;

  //
  // Initialize all entries in array of length N to the value '9'
  //
  int vec[N] = {9};

  // Create a bitwise or reduction object with initial value of '5'
  RAJA::ReduceBitOr< RAJA::omp_reduce, int > my_or(5);

  // Run a kernel using the reduction object
  RAJA::forall<RAJA::omp_parallel_for_exec>( RAJA::RangeSegment(0, N),
    [=](RAJA::Index_type i) {

    my_or |= vec[i];

  });

  // After kernel is run, extract the reduced value
  int my_or_reduce_val = static_cast<int>(my_or.get());

The result of the reduction is the value '13'. In binary representation
(i.e., bits), :math:`9 = ...01001` (the vector entries) and 
:math:`5 = ...00101` (the initial reduction value). 
So :math:`9 | 5 = ...01001 | ...00101 = ...01101 = 13`.

-------------------
Reduction Policies
-------------------

For more information about available RAJA reduction policies and guidance
on which to use with RAJA execution policies, please see 
:ref:`reducepolicy-label`.

--------------------------------
Experimental Reduction Interface
--------------------------------

An experimental reduction interface is now available that hopes to improve
upon the current reduction model in RAJA. This interface allows ``RAJA::forall``
to take optional "plugin-like" objects to extend the execution behaviour 
of a ``RAJA::forall`` execution context.


RAJA::expt::Reduce
..................
::

  double rs;
  double rm;
      
  RAJA::forall<EXEC_POL> ( Res, Seg, 
  RAJA::expt::Reduce<RAJA::operators::plus>(&rs),
  RAJA::expt::Reduce<RAJA::operators::minimum>(&rm),
  [=] (int i, double& _rs, double& _rm) {
    _rs += ...
    _rm = RAJA_MIN(..., _rm); 
  }
  );
  
  std::cout << rs ...
  std::cout << rm ...

* ``RAJA::expt::Reduce`` takes a target variable to write the final result to 
  (``rs``, ``rm``).
* It passes a corresponding argument to the RAJA lambda to be used as the 
  local instance of the target(``_rs``, ``_rm``).
* The local variable is initialized with the "identity" of the reduction 
  operation to be performed.
* A reduction is performed implicitly by the ``RAJA::forall`` across thread 
  copies of the local variable.
* Finally, the reduction operation is performed against the original value of 
  the target and the result of the reduction.
* The final value can be returned simply be referencing the target variable.

RAJA::expt::ValLoc
..................

RAJA supports ``Loc`` reductions. With this new interface ``Loc`` reductions 
can be performed using ``ValLoc<T>`` types. Since they are strongly typed they
provide min() and max() operations. Users must also use getVal() and getLoc to
return results.
::

  using VL_INT = RAJA::expt::ValLoc<int>;
  VL_INT rm_loc;
      
  RAJA::forall<EXEC_POL> ( Res, Seg, 
  RAJA::expt::Reduce<RAJA::operators::minimum>(&rm_loc),
  [=] (int i, VL_INT& _rm_loc) {
    _rm_loc = RAJA_MIN(..., _rm_loc);  
  }
  );

  std::cout << rm_loc.getVal() ...
  std::cout << rm_loc.getLoc() ...

Lambda Arguments
................

This interface takes advantage of C++ parameter packs to allow users to define
any number of ``expt::Reduce`` objects in their RAJA::forall calls.
::

  using VL_INT = RAJA::expt::ValLoc<int>;
  VL_INT rm_loc;
  double rs;
  double rm;
        
  RAJA::forall<EXEC_POL> ( Res, Seg, 
    RAJA::expt::Reduce<RAJA::operators::plus>(&rs),        // --> 1 double added
    RAJA::expt::Reduce<RAJA::operators::minimum>(&rm),     // --> 1 double added
    RAJA::expt::Reduce<RAJA::operators::minimum>(&rm_loc), // --> 1 VL_INT added
    RAJA::expt::KernelName("MyFirstRAJAKernel"),           // --> NO args added
    [=] (int i, double& _rs, double& _rm, VL_INT& _rm_loc) {
      _rs += ...
      _rm = RAJA_MIN(..., _rm); 
      _rm_loc.min(...);
    }
  );

  std::cout << rs ...
  std::cout << rm ...
  std::cout << rm_loc.getVal() ...
  std::cout << rm_loc.getLoc() ...

The lambda arguments are passed in the same respective order to that of the
ForallParams. Both the types and number of arguments are required to be correct
in order to compile successfully otherwise a static assertion will be triggered:
::

  LAMBDA Not invocable w/ EXPECTED_ARGS.

.. note:: This static assert is only enabled when passing a generic lambda, 
          this check will not happed when passing extended-lambdas (i.e. DEVICE
          tagged lambdas) or other functor like objects.


