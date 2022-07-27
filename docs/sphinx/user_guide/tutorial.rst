.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _tutorial-label:

**********************
RAJA Tutorial
**********************

This section contains a self-paced tutorial that shows how to use most RAJA
features by way of a sequence of examples. Complete working codes for 
the examples are located in the ``RAJA/examples`` directory. You are 
encouraged to build and run them, as well as modify them to try out different
variations. 

We also maintain a repository of tutorial slide presentations 
`RAJA Tutorials Repo <https://github.com/LLNL/RAJA-tutorials>`_ which we use 
when we give in-person or virtual online tutorials in various venues. The 
presentations complement the material found here. The tutorial material 
evolves as we add new features to RAJA, so refer to it periodically if you 
are interested in learning about new things in RAJA.

To understand the GPU examples (e.g., CUDA), it is also important to know the 
difference between CPU (host) and GPU (device) memory allocations and how 
transfers between those memory spaces work. For a detailed discussion, see 
`Device Memory <http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory>`_. 

RAJA does not provide a memory model. This is by design as developers of many
of applications that use RAJA prefer to manage memory themselves. Thus, users 
are responsible for ensuring that data is properly allocated and initialized 
on a GPU device when running GPU code. This can be done using explicit host 
and device allocation and copying between host and device memory spaces or via 
unified memory (UM), if available. The RAJA Portability Suite contains other
libraries, such as `CHAI <https://github.com/LLNL/CHAI>`_ and 
`Umpire <https://github.com/LLNL/Umpire>`_, that complement RAJA by 
providing alternatives to manual programming model specific memory operations.

.. _tutorial-lambda-label:

===============================
A Little C++ Background
===============================

To understand the discussion and code examples, a working knowledge of C++ 
templates and lambda expressions is required. So, before we begin, we provide 
a bit of background discussion of basic aspects of how RAJA use employs C++ 
templates and lambda expressions, which is essential to using RAJA successfully.

RAJA makes heavy use of C++ templates and using RAJA most easily and 
effectively is done by representing the bodies of loop kernels as C++ lambda 
expressions. Alternatively, C++ functors can be used, but they make 
application source code more complex, potentially placing a significant 
negative burden on source code readability and maintainability.

-----------------------------------
C++ Templates
-----------------------------------

C++ templates enable one to write generic code and have the compiler generate 
a specific implementation for each set of template parameter types specified.
For example, the ``RAJA::forall`` method to execute loop kernels is a 
template method defined as::

  template <typename ExecPol,
            typename IdxType,
            typename LoopBody>
  forall(IdxType&& idx, LoopBody&& body) {
     ...
  }

Here, "ExecPol", "IdxType", and "LoopBody" are C++ types a user specifies in
their code so they are seen by the compiler when the code is built. 
For example::

  RAJA::forall< RAJA::seq_exec >( RAJA::RangeSegment(0, N), [=](int i) {
    a[i] = b[i] + c[i];
  });

Here, the execution policy type ``RAJA::seq_exec`` is an explicit template
argument parameter used to choose as specific implementation of the 
``RAJA::forall`` method. The ``IdxType`` and ``LoopBody`` types are deduced by 
the compiler based on what arguments are passed to the ``RAJA::forall`` method;
i.e., the ``IdxType`` is the stride-1 range::

  RAJA::RangeSegment(0, N)

and the ``LoopBody`` type is the lambda expression::

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

There are several issues to note about using C++ lambda expressions to 
represent kernel bodies with RAJA. We describe them here.

 * **Prefer by-value lambda capture.** 

   We recommend `capture by-value` for all lambda kernel bodies passed to 
   RAJA execution methods. To execute a RAJA loop on a non-CPU device, such 
   as a GPU, all variables accessed in the loop body must be passed into the 
   GPU device data environment. Using capture by-value for all RAJA-based 
   lambda usage will allow your code to be portable for either CPU or GPU 
   execution. In addition, the read-only nature of variables captured 
   by-value can help avoid incorrect CPU code since the compiler will report 
   incorrect usage.

|br|

 * **The  'device' annotation is required for device execution using CUDA or HIP.** 

   Any lambda passed to a CUDA or HIP execution context (or function called from a
   device kernel, for that matter) must be decorated with 
   the ``__device__`` annotation; for example::
     
     RAJA::forall<RAJA::cuda_exec<BLOCK_SIZE>>( range, [=] __device__ (int i) { ... } );

   Without this, the code will not compile and generate compiler errors
   indicating that a 'host' lambda cannot be called in 'device' code.

   RAJA provides the macro ``RAJA_DEVICE`` that can be used to help switch
   between host-only or device-only compilation.
    
|br|

 * **Use 'host-device' annotation on a lambda carefully.**

   RAJA provides the macro ``RAJA_HOST_DEVICE`` to support the dual
   annotation ``__ host__ __device__``, which makes a lambda or function
   callable from CPU or CUDA device code. However, when CPU performance is 
   important, **the host-device annotation should be applied carefully on a 
   lambda that is used in a host (i.e., CPU) execution context**. Although
   compiler improvements in recent years (esp. nvcc) have signficantly
   improved support for host-device lambda expressions, a loop kernel 
   containing a lambda annotated in this way may run noticeably slower on 
   a CPU than the same lambda with no annotation depending on the version of 
   the compiler you are using. To be sure that your code is not suffering
   a performance issue, we recommend comparing CPU execution timings of 
   important kernels with and without annotations.
    
|br|

 * **Cannot use 'break' and 'continue' statements in a lambda.** 

   In this regard, a lambda expression is similar to a function. So, if you 
   have loops in your code with these statements, they should be rewritten. 
    
|br|

 * **Global variables are not captured in a lambda.** 

   This fact is due to the C++ standard. If you need (read-only) access to a 
   global variable inside a lambda expression, one solution is to make a local 
   reference to it; for example::

     double& ref_to_global_val = global_val;

     RAJA::forall<RAJA::cuda_exec<BLOCK_SIZE>>( range, [=] __device__ (int i) { 
       // use ref_to_global_val
     } );
    
|br|

 * **Local stack arrays may not be captured by CUDA device lambdas.** 

   Although this is inconsistent with the C++ standard (local stack arrays
   are properly captured in lambdas for code that will execute on a CPU), 
   attempting to access elements in a local stack array in a CUDA device 
   lambda may generate a compilation error depending on the version of the 
   device compiler you are using. One solution to this problem is to wrap the 
   array in a struct; for example::

     struct array_wrapper {
       int[4] array;
     } bounds;

     bounds.array = { 0, 1, 8, 9 };

     RAJA::forall<RAJA::cuda_exec<BLOCK_SIZE>>(range, [=] __device__ (int i) {
       // access entries of bounds.array
     } );

   This issue was resolved in the 10.1 release of nvcc. If you 
   are using an earlier version of nvcc, an implementation
   similar to the one above will be required. 
    
    
================
RAJA Examples
================

The remainder of this tutorial illustrates how to use RAJA features with
working code examples that are located in  the ``RAJA/examples`` 
directory. Additional information about the RAJA features 
used can be found in :ref:`features-label`.

The examples demonstrate CPU execution (sequential, OpenMP
multithreading) and GPU execution (CUDA and/or HIP). Examples that show how 
to use RAJA with other parallel programming model back-ends that are in 
development will appear in future RAJA releases. For adventurous users who 
wish to try experimental features, usage is similar to what is shown in the 
examples here.

All RAJA programming model support features are enabled via CMake options,
which are described in :ref:`configopt-label`. 

For the purposes of discussion of each example, we assume that any and all 
data used has been properly allocated and initialized. This is done in the 
example code files, but is not discussed further here.

Finally, RAJA kernel variants in the examples illustrate how a kernel can be 
run with different programming model back-ends by simply changing an 
execution policy type. RAJA application users typically define type aliases 
for execution policies in header files so that these types can be easily 
changed, and the code can be compiled to run differently, without changing 
any loop kernel source code. Another benefit of this approach is that such 
type changes are easily propagated to many kernels with a change to 
a single file. However, in the example codes, we make all execution policy 
types explicit for clarity.

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
   tutorial/indexset_segments.rst
   tutorial/vertexsum_coloring.rst
   tutorial/dot_product.rst
   tutorial/reductions.rst
   tutorial/atomic_histogram.rst
   tutorial/scan.rst
   tutorial/sort.rst

.. _tutorialcomplex-label:

=================================================================
Complex Loops and Advanced RAJA Features
=================================================================

RAJA provides two APIs for expressing complex loop kernels, such as nested
loops: ``RAJA::kernel`` and ``RAJA::launch`` 
(often referred to as "RAJA Teams").

``RAJA::kernel``
is analogous to ``RAJA::forall`` in that the semantics involve kernel execution
templates, execution policies, iteration spaces, and lambda kernel bodies.
The main differences with ``RAJA::forall`` are that ``RAJA::kernel`` execution
policies can be much more complicated, ``RAJA::kernel`` requires a tuple
of iteration spaces (one for each level in a loop nest), and ``RAJA::kernel``
can accept multiple lambda expressions to express parts of a kernel body.
Almost all aspects of kernel execution are represented in the execution 
policies, which support a wide range of compile-time loop transformations and
advanced features. 

``RAJA::launch``, in contrast, uses the 
``RAJA::launch`` template, which takes a ``RAJA::Grid`` type argument for 
expressing the teams-thread lauch configuration, and a lambda expression
which takes a ``RAJA::LaunchContext`` argument. The lambda provides an
execution environment (e.g., CPU or GPU) for a kernel. Within that 
environment users execute kernel operations using ``RAJA::loop<EXEC_POL>``
method calls, which take lambda expressions to express loop details.

Which RAJA API to use depends on personal preference (kernel structure
is more explicit in application source code with ``RAJA::launch``, and more
concise and arguably more opaque with ``RAJA::kernel``), and other concerns,
such as portability requirements, runtime policy selection, etc.
There is a large overlap of algorithms that can be expressed using either
API, and there are things that one can do with one or the other but not both.

In the following sections, we introduce the basic mechanics and features
of both APIs with examples and exercises. We also present a sequence of 
matrix-matrix multiplication examples using both APIs to compare and contrast.
 
===========================================================================
``RAJA::kernel`` Based Loops: Nested loops with complex execution policies 
===========================================================================

The examples in this section illustrate various features of the 
``RAJA::kernel`` API used to execute nested loop kernels. It describes how to 
construct kernel execution policies, use different view types and tiling 
mechanisms to transform loop patterns. More informatrion can be found in
:ref:`loop_elements-kernel-label`.

.. toctree::
   :maxdepth: 1

   tutorial/nested_loop_reorder.rst

=================================================================
Team based Loops: Nested loops with a team/thread model
=================================================================

The examples in this section illustrate how to use ``RAJA::expt::launch``
to create an run-time selectable execution space for expressing algorithms
in terms of threads and teams.

.. |br| raw:: html

   <br />

.. toctree::
   :maxdepth: 1

   tutorial/teams_basic.rst
   tutorial/naming_kernels.rst

==============================
Other Advanced RAJA Features
==============================

.. toctree::
   :maxdepth: 1

   tutorial/halo-exchange.rst
   tutorial/matrix_multiply.rst
   tutorial/permuted-layout.rst
   tutorial/offset-layout.rst
   tutorial/tiled_matrix_transpose.rst
   tutorial/matrix_transpose_local_array.rst
   tutorial/halo-exchange.rst
