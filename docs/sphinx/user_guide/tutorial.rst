.. ##
.. ## Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/COPYRIGHT file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _tutorial-label:

**********************
RAJA Tutorial
**********************

In addition to the tutorial portion of this RAJA User Guide, we maintain
a repository of tutorial presntation materials here `RAJA Tutorials Repo <https://github.com/LLNL/RAJA-tutorials>`_.

This RAJA tutorial introduces RAJA concepts and capabilities via a 
sequence of examples of increasing complexity. Complete working codes for 
the examples are located in the ``RAJA``examples`` directory. The RAJA 
tutorial evolves as we add new features to RAJA, so refer to it periodically
if you are interested in learning about them.

To understand the discussion and code examples, a working knowledge of C++ 
templates and lambda expressions is required. So, before we begin, we provide 
a bit of background discussion of basic aspects of how RAJA use employs C++ 
templates and lambda expressions, which is essential to using RAJA successfully.

To understand the GPU examples (e.g., CUDA), it is also important to know the 
difference between CPU (host) and GPU (device) memory allocations and how 
transfers between those memory spaces work. For a detailed discussion, see 
`Device Memory <http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory>`_. 

RAJA does not provide a memory model. This is by design as developers of many
of applications that use RAJA prefer to manage memory themselves. Thus, users 
are responsible for ensuring that data is properly allocated and initialized 
on a GPU device when running GPU code. This can be done using explicit host 
and device allocation and copying between host and device memory spaces or via 
unified memory (UM), if available. RAJA developers also support a library 
called `CHAI <https://github.com/LLNL/CHAI>`_ which complements RAJA by 
providing a alternative to manual host-device memory copy calls or UM. 
For more information, see :ref:`plugins-label`.

.. _tutorial-lambda-label:

===============================
A Little C++ Background
===============================

RAJA makes heavy use of C++ templates and using RAJA most easily and 
effectively is done by representing the bodies of loop kernels as C++ lambda 
expressions. Alternatively, C++ factors can be used, but they make 
application source code more complex, potentially placing a significant 
negative burden on source code readability and maintainability.

-----------------------------------
C++ Templates
-----------------------------------

C++ templates enable one to write generic code and have the compiler generate 
a specific implementation for each set of template parameter types you use.
For example, the ``RAJA::forall`` method to execute loop kernels is a 
template method defined as::

  template <typename ExecPol,
            typename IdxType,
            typename LoopBody>
  forall(IdxType&& idx, LoopBody&& body) {
     ...
  }

Here, "ExecPol", "IdxType", and "LoopBody" are C++ types a user specifies in
their code; for example::

  RAJA::forall< RAJA::seq_exec >( RAJA::RangeSegment(0, N), [=](int i) {
    a[i] = b[i] + c[i];
  });

The "IdxType" and "LoopBody" types are deduced by the compiler based on what 
arguments are passed to the ``RAJA::forall`` method. Here, the loop body type 
is defined by the lambda expression::

  [=](int i) { a[i] = b[i] + c[i]; }

-----------------------------------
Elements of C++ Lambda Expressions
-----------------------------------

Here, we provide a brief description of the basic elements of C++ lambda
expressions. A more technical and detailed discussion is available here:
`Lambda Functions in C++11 - the Definitive Guide <https://www.cprogramming.com/c++11/c++11-lambda-closures.html>`_ 

Lambda expressions were introduced in C++ 11 to provide a lexical-scoped 
name binding; specifically, a *closure* that stores a function with a data 
environment. That is, a lambda expression can *capture* variables from an 
enclosing scope for use within the local scope of the function expression.

A C++ lambda expression has the following form::

  [capture list] (parameter list) {function body}

The ``capture list`` specifies how variables outside the lambda scope are pulled
into the lambda data environment. The ``parameter list`` defines arguments 
passed to the lambda function body -- for the most part, lambda arguments
are just like arguments in a regular C++ method. Variables in the capture list 
are initialized when the lambda expression is created, while those in the 
parameter list are set when the lambda expression is called. The body of a 
lambda expression is similar to the body of an ordinary C++ method.
RAJA templates, such as ``RAJA::forall`` and ``RAJA::kernel`` pass arguments 
to lambdas based on usage and context; e.g., loop iteration indices.

A C++ lambda expression can capture variables in the capture list by value 
or by reference. This is similar to how arguments to C++ methods are passed; 
i.e., *pass-by-reference* or *pass-by-value*. However, there are some subtle 
differences between lambda variable capture rules and those for ordinary
methods. Variables mentioned in the capture list with no extra symbols are 
captured by value. Capture-by-reference is accomplished by using the 
reference symbol '&' before the variable name; for example::

  int x;
  int y = 100;
  [&x, &y](){ x = y; };

generates a lambda expression that captures both 'x' and 'y' by reference 
and assigns the value of 'y' to 'x' when called. The same outcome would be 
achieved by writing::

  [&](){ x = y; };   // capture all lambda arguments by reference...

or::

  [=, &x](){ x = y; };  // capture 'x' by reference and 'y' by value...  

Note that the following two attempts will generate compilation errors::

  [=](){ x = y; };      // error: all lambda arguments captured by value,
                        //        so cannot assign to 'x'.
  [x, &y](){ x = y; };  // error: cannot assign to 'x' since it is captured
                        //        by value.

**Specifically, a variable hat is captured by value is read-only.**

----------------------------------------
A Few Notes About Lambda Usage With RAJA 
----------------------------------------

There are several issues to note about C++ lambda expressions; in particular, 
with respect to RAJA usage. We describe them here.

 * **Prefer by-value lambda capture.** 

   We recommended `capture by-value` for all lambda loop bodies passed to 
   RAJA execution methods. To execute a RAJA loop on a non-CPU device, such 
   as a GPU, all variables accessed in the loop body must be passed into the 
   GPU device data environment. Using capture by-value for all RAJA-based 
   lambda usage will allow your code to be portable for either CPU or GPU 
   execution. In addition, the read-only nature of variables captured 
   by-value can help avoid incorrect CPU code since the compiler will report 
   incorrect usage.


 * **Must use 'device' annotation for CUDA device execution.** 

   Any lambda passed to a CUDA execution context (or function called from a
   CUDA device kernel, for that matter) must be decorated with 
   the ``__device__`` annotation; for example::
     
     RAJA::forall<RAJA::cuda_exec>( range, [=] __device__ (int i) { ... } );

   Without this, the code will not compile and generate compiler errors
   indicating that a 'host' lambda cannot be called from 'device' code.

   RAJA provides the macro ``RAJA_DEVICE`` that can be used to help switch
   between host-only or device-only CUDA compilation.
    

 * **Avoid 'host-device' annotation on a lambda that will run in host code.**

   RAJA provides the macro ``RAJA_HOST_DEVICE`` to support the dual
   CUDA annotation ``__ host__ __device__``. This makes a lambda or function
   callable from CPU or CUDA device code. However, when CPU performance is 
   important, **the host-device annotation should be applied carefully on a 
   lambda that is used in a host (i.e., CPU) execution context**. 
   Unfortunately, a loop kernel containing a lambda annotated in this way 
   may run noticeably slower on a CPU than the same lambda with no annotation 
   depending on the version of the nvcc compiler you are using.
    

 * **Cannot use 'break' and 'continue' statements in a lambda.** 

   In this regard, a lambda expression is similar to a function. So, if you 
   have loops in your code with these statements, they should be rewritten. 
    

 * **Global variables are not captured in a lambda.** 

   This fact is due to the C++ standard. If you need (read-only) access to a 
   global variable inside a lambda expression, one solution is to make a local 
   reference to it; for example::

     double& ref_to_global_val = global_val;

     RAJA::forall<RAJA::cuda_exec>( range, [=] __device__ (int i) { 
       // use ref_to_global_val
     } );
    

 * **Local stack arrays may not be captured by CUDA device lambdas.** 

   Although this is inconsistent with the C++ standard (local stack arrays
   are properly captured in lambdas for code that will execute on a CPU), 
   attempting to access elements in a local stack array in a CUDA device 
   lambda may generate a compilation error depending on the version of the 
   nvcc compiler you are using. One solution to this problem is to wrap the 
   array in a struct; for example::

     struct array_wrapper {
       int[4] array;
     } bounds;

     bounds.array = { 0, 1, 8, 9 };

     RAJA::forall<RAJA::cuda_exec>(range, [=] __device__ (int i) {
       // access entries of bounds.array
     } );

   This issue appears to be resolved in in the 10.1 release of CUDA. If you 
   are using an earlier version of nvcc, an implementation
   similar to the one above will be required. 
    
    
================
RAJA Examples
================

The remainder of this tutorial illustrates how to use RAJA features with
working code examples that are located in  the ``RAJA/examples`` 
directory. Additional information about the RAJA features 
used can be found in :ref:`features-label`.

The examples demonstrate CPU execution (sequential, SIMD, OpenMP
multithreading) and CUDA GPU execution. Examples that show how to use
RAJA with other parallel programming model back-ends that are in 
development will appear in future RAJA releases. For adventurous users who 
wish to try experimental features, usage is similar to what is shown in the 
examples here.

All RAJA programming model support features are enabled via CMake options,
which are described in :ref:`configopt-label`. 

For the purposes of discussion of each example, we assume that any and all 
data used has been properly allocated and initialized. This is done in the 
example code files, but is not discussed further here.

.. _tutorialbasic-label:

=====================================
Simple Loops and Basic RAJA Features
=====================================

The examples in this section illustrate how to use ``RAJA::forall`` methods
to execute simple loop kernels; i.e., non-nested loops. It also describes
iteration spaces, reductions, atomic operations, scans, and sorts.

.. toctree::
   :maxdepth: 1

   tutorial/add_vectors.rst
   tutorial/dot_product.rst
   tutorial/indexset_segments.rst
   tutorial/vertexsum_coloring.rst
   tutorial/reductions.rst
   tutorial/atomic_histogram.rst
   tutorial/scan.rst
   tutorial/sort.rst

.. _tutorialcomplex-label:

=================================================================
Complex Loops: Transformations and Advanced RAJA Features
=================================================================

The examples in this section illustrate how to use ``RAJA::kernel`` methods
to execute complex loop kernels, such as nested loops. It also describes
how to construct kernel execution policies, use different view types and
tiling mechanisms to transform loop patterns.

.. toctree::
   :maxdepth: 1

   tutorial/matrix_multiply.rst
   tutorial/nested_loop_reorder.rst
   tutorial/permuted-layout.rst
   tutorial/offset-layout.rst
   tutorial/tiled_matrix_transpose.rst
   tutorial/matrix_transpose_local_array.rst
   tutorial/halo-exchange.rst
