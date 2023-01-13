.. ##
.. ## Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
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

An experimental reduction interface is now available that offers several 
usability and performance advantages over the current reduction model in RAJA. 
The new interface allows ``RAJA::forall`` to take optional "plugin-like" 
objects to extend the execution behavior of a ``RAJA::forall`` execution 
context.

The new interface passes ``RAJA::expt::Reduce<OP_TYPE>`` objects as function 
arguments to ``RAJA::forall`` and provides users with thread-local variables
of the reduction data type to be updated inside the lambda. This differs 
from the current reduction model in which ``RAJA::ReduceOP<REDUCE_POL, T>``
objects are captured by the user-supplied kernel body lambda expression.


RAJA::expt::Reduce
..................
::

  double* a = ...;

  double rs = 0.0;
  double rm = 1e100;
      
  RAJA::forall<EXEC_POL> ( Res, Seg, 
  RAJA::expt::Reduce<RAJA::operators::plus>(&rs),
  RAJA::expt::Reduce<RAJA::operators::minimum>(&rm),
  [=] (int i, double& _rs, double& _rm) {
    _rs += a[i];
    _rm = RAJA_MIN(a[i], _rm); 
  }
  );
  
  std::cout << rs ...
  std::cout << rm ...

* Each ``RAJA::expt::Reduce`` argument to ``RAJA::forall`` is templated on
  a reduction operator, and takes a pointer to a target variable to write 
  the final reduction result to, ``&rs`` and ``&rm`` in the example code
  above. The reduction operation will include the existing value of
  the given target variable.
* The kernel body lambda expression passed to ``RAJA::forall`` must have a 
  parameter corresponding to each ``RAJA::expt::Reduce`` argument, ``_rs`` and 
  ``_rm`` in the example code. These parameters refer to a local target for each
  reduction operation. It is important to note that the parameters follow the
  kernel iteration variable, ``i`` in this case, and appear in the same order as the 
  corresponding ``RAJA::expt::Reduce`` arguments to ``RAJA::forall``. The
  parameters' types must be references to the types used in the
  ``RAJA::expt::Reduce`` arguments.
* The local variables referred to by ``_rs`` and ``_rm`` are initialized with the
  *identity* of the reduction operation to be performed.
* The local variables are updated in the user supplied lambda.
* The local variables are reduced to a single value, combining their values across all
  threads participating in the ``RAJA::forall`` execution.
* Finally, the target variable is updated with the result of the ``RAJA::forall`` reduction
  by performing the reduction operation to combine the existing value of the target 
  variable and the result of the ``RAJA::forall`` reduction.
* The final reduction value is accessed by referencing the target variable 
  passed to ``RAJA::expt::Reduce`` in the ``RAJA::forall`` method.

.. note:: In the above example ``Res`` is a resource object that must be 
          compatible with the ``EXEC_POL``. ``Seg`` is the iteration space
          object for ``RAJA::forall``.

.. important:: The order and types of the local reduction variables in the
               kernel body lambda expression must match exactly with the
               corresponding ``RAJA::expt::Reduce`` arguments to the
               ``RAJA::forall`` to ensure that the correct result is obtained.

RAJA::expt::ValLoc
..................

As with the current RAJA reduction interface, the new interface supports *loc* 
reductions, which provide the ability to get a kernel/loop index at which the 
final reduction value was found. With this new interface, *loc* reductions 
are performed using ``ValLoc<T>`` types. Since they are strongly typed, they
provide ``min()`` and ``max()`` operations that are equivalent to using 
``RAJA_MIN()`` or ``RAJA_MAX`` macros as demonstrated in the code example below.
Users must use the ``getVal()`` and ``getLoc()`` methods to access the reduction 
results::

  double* a = ...;

  using VL_DOUBLE = RAJA::expt::ValLoc<double>;
  VL_DOUBLE rm_loc;
      
  RAJA::forall<EXEC_POL> ( Res, Seg, 
  RAJA::expt::Reduce<RAJA::operators::minimum>(&rm_loc),
  [=] (int i, VL_DOUBLE& _rm_loc) {
    _rm_loc = RAJA_MIN(VL_DOUBLE(a[i], i), _rm_loc);  
    //_rm_loc.min(VL_DOUBLE(a[i], i)); // Alternative to RAJA_MIN
  }
  );

  std::cout << rm_loc.getVal() ...
  std::cout << rm_loc.getLoc() ...

Lambda Arguments
................

This interface takes advantage of C++ parameter packs to allow users to pass
any number of ``RAJA::expt::Reduce`` objects to the ``RAJA::forall`` method::

  double* a = ...;

  using VL_DOUBLE = RAJA::expt::ValLoc<double>;
  VL_DOUBLE rm_loc;
  double rs;
  double rm;
        
  RAJA::forall<EXEC_POL> ( Res, Seg, 
    RAJA::expt::Reduce<RAJA::operators::plus>(&rs),        // --> 1 double added
    RAJA::expt::Reduce<RAJA::operators::minimum>(&rm),     // --> 1 double added
    RAJA::expt::Reduce<RAJA::operators::minimum>(&rm_loc), // --> 1 VL_DOUBLE added
    RAJA::expt::KernelName("MyFirstRAJAKernel"),           // --> NO args added
    [=] (int i, double& _rs, double& _rm, VL_DOUBLE& _rm_loc) {
      _rs += a[i];
      _rm = RAJA_MIN(a[i], _rm); 
      _rm_loc.min(VL_DOUBLE(a[i], i));
    }
  );

  std::cout << rs ...
  std::cout << rm ...
  std::cout << rm_loc.getVal() ...
  std::cout << rm_loc.getLoc() ...

Again, the lambda expression parameters are in the same order as
the ``RAJA::expt::Reduce`` arguments to ``RAJA::forall``. Both the types and 
order of the parameters must match to get correct results and to compile 
successfully. Otherwise, a static assertion will be triggered::

  LAMBDA Not invocable w/ EXPECTED_ARGS.

.. note:: This static assert is only enabled when passing an undecorated C++
          lambda. Meaning, this check will not happen when passing 
          extended-lambdas (i.e. DEVICE tagged lambdas) or other functor like
          objects.

.. note:: The experimental ``RAJA::forall`` interface is more flexible than the
          current implementation, other optional arguments besides
          ``RAJA::expt::Reduce`` can be passed to a ``RAJA::forall`` to extend
          its behavior. In the above example we demonstrate using 
          ``RAJA::expt::KernelName``, which wraps a ``RAJA::forall`` executing 
          under a ``HIP`` or ``CUDA`` policy in a named region. Use of 
          ``RAJA::expt::KernelName`` does not require an additional
          parameter in the lambda expression.
