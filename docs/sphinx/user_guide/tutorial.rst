.. ##
.. ## Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
.. ##
.. ## Produced at the Lawrence Livermore National Laboratory
.. ##
.. ## LLNL-CODE-689114
.. ##
.. ## All rights reserved.
.. ##
.. ## This file is part of RAJA.
.. ##
.. ## For details about use and distribution, please read RAJA/LICENSE.
.. ##

.. _tutorial-label:

**********************
RAJA Tutorial
**********************

This RAJA tutorial introduces the most commonly-used RAJA concepts and
capabilities via a sequence of simple examples. 

To understand the discussion and example codes, a working knowledge of C++ 
templates and lambda functions is required. Here, we provide a bit 
of background discussion of the key aspect of C++ lambda expressions, which 
are essential to using RAJA easily.

It is also important to understand the difference between CPU (host) and 
GPU (device) memory allocations and transfers work. For a detailed discussion, 
see `Device Memory <http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory>`_. RAJA does not provide a memory model by design. So users 
are responsible for ensuring that data is properly allocated and initialized 
on the device when working running GPU code. This can be done using explicit 
host and device allocation and copying between host and device memory spaces
or via CUDA unified memory (UM), if available. RAJA developers also support a 
library called ``CHAI`` which is complementary to RAJA and which provides a 
simple alternative to manual CUDA calls or UM. For more information about 
CHAI, see :ref:`plugins-label`.

.. _tutorial-lambda-label:

===============================
A Little C++ Lambda Background
===============================

RAJA is used most easily and effectively by employing C++ lambda expressions
for the bodies of loop kernels. Alternatively, C++ functors can be used, but
we don't recommend them as they have a significant negative impact on source
code readability, potentially.

-----------------------------------
Elements of C++ Lambda Expressions
-----------------------------------

Here, we provide a brief description of the basic elements of C++ lambda
expressions. A more technical and detailed discussion is available here:
`Lambda Functions in C++11 - the Definitive Guide <https://www.cprogramming.com/c++11/c++11-lambda-closures.html>`_ 

Lambda expressions were introduced in C++ 11 to provide a lexically-scoped 
name binding; i.e., a *closure* that stores a function with a data environment.
In particular, a lambda has the ability to *capture* variables from an 
enclosing scope for use within the local scope of the lambda expression.

A C++ lambda expression has the following form::

  [capture list] (parameter list) {function body}

The ``capture list`` specifies how variables outside the lambda scope are pulled
into the lambda data environment. The ``parameter list`` defines arguments 
passed to the lambda function body -- for the most part, lambda arguments
are just like arguments for a standard C++ method. Values in the capture list 
are initialized when the lambda is created, while values in the parameter list 
are set when the lambda function is called. The body of a lambda function is
similar to the body of an ordinary C++ method.
RAJA template methods pass arguments to lambdas based on usage and context;
typically, these are loop indices.

A C++ lambda can capture values in the capture list by value or by reference.
This is similar to how arguments to C++ methods are passed; e.g., 
pass-by-reference or pass-by-value. However, there are some subtle 
differences between lambda variable capture rules and those for ordinary
methods. Variables mentioned in the capture list with no extra symbols are 
captured by value. Capture-by-reference is accomplished by using the 
reference symbol '&' before the variable name; for example::

  int x;
  int y = 100;
  [&x, &y](){ x = y; };

generates a lambda that captures 'x' and 'y' by reference and assigns the 
value of 'y' to 'x' when called. The same outcome would be achieved by 
writing::

  [&](){ x = y; };   // capture all lambda arguments by reference...

or::

  [=, &x](){ x = y; };  // capture 'x' by reference and 'y' by value...  

Note that the following two attempts will generate compilation errors::

  [=](){ x = y; };      // capture all lambda arguments by value...
  [x, &y](){ x = y; };  // capture 'x' by value and all other arguments
                        // (e.g., 'y') by reference...

It is illegal to assign a value to a variable 'x' that is captured
by value since it is `read-only`.

-----------------------------------
C++ Lambda Concerns
-----------------------------------

There are several issues to note about C++ lambdas; in particular, 
with respect to RAJA usage. We describe them here.

 * **Prefer by-value lambda capture.** We recommended using `capture by-value` 
   for all lambda loop bodies passed to RAJA execution methods. To execute a
   RAJA loop on a non-CPU device, such as a GPU, all variables accessed in
   the loop body must be passed into the device data environment. Using 
   capture by-value for all RAJA-based lambda usage will allow your
   code to be portable for either CPU or GPU execution. In addition, 
   the read-only nature of variables captured by-value can help avoid 
   incorrect CPU code via compiler errors.

 * **Need lambda 'device' annotation for CUDA device execution.** 
   Any lambda passed to a CUDA execution context (or function called from a
   CUDA device kernel, for that matter) must be annotated with 
   the ``__device__`` annotation; e.g.,::
     
     RAJA::forall<RAJA::cuda_exec>( range, [=] __device__ (int i) { ... } );

   Without this, the code will not compile and generate compiler errors
   indicating that a 'host' lambda cannot be called from 'device' code.

   RAJA provides the macro ``RAJA_DEVICE`` that can be used to help switch
   between host-only or device-only CUDA compilation.

 * **Avoid 'host-device' annotation on a lambda that will run in host code.**
   RAJA provides the macro ``RAJA_HOST_DEVICE`` to support the dual
   CUDA annotation ``__ host__ __device__``. This makes a lambda or function
   callable from CPU or CUDA device code. However, when CPU performance is 
   important, **the host-device annotation should not be used on a lambda that
   is used in a host (i.e., CPU) execution context**. Unfortunately, a loop 
   kernel containing a lambda annotated in this way will run noticeably 
   slower on a CPU than the same lambda with no annotation.

 * **Cannot use 'break' and 'continue' statements in a lambda.** In this 
   regard, a lambda is similar to a function. So, if you have loops in your
   code with these statements, they must be rewritten. 

 * **Global variables are not captured in a lambda.** This fact is due to the
   C++ standard. If you need (read-only) access to a global variable inside
   a lambda expression, make a local reference to it; e.g.,::

     double& ref_to_global_val = global_val;

     RAJA::forall<RAJA::cuda_exec>( range, [=] __device__ (int i) { 
       // use ref_to_global_val
     } );

 * **Local stack arrays are not captured by CUDA device lambdas.** Although
   this is inconsistent with the C++ standard, attempting to access elements
   in a local stack array in a CUDA device lambda is a compilation error.
   A solution to this is to wrap the array in a struct; e.g.,::

     struct array_wrapper {
       int[4] array;
     } bounds;

     bounds.array = { 0, 1, 8, 9 };

     RAJA::forall<RAJA::cuda_exec>(range, [=] __device__ (int i) {
       // access entries of bounds.array
     } );
    
================
RAJA Examples
================

The remainder of this tutorial illustrates how to exercise various RAJA 
features using simple examples. Additional information about the RAJA
features used can be found in the different sections of :ref:`features-label`.

The examples demonstrate CPU execution (sequential, SIMD, OpenMP
multi-threading) and CUDA GPU execution. RAJA supports other parallel 
programming model back-ends such as Intel Threading Building Blocks 
(TBB), OpenMP target offload, etc. and support for additional models
is also under development. RAJA usage of other programming models,
not shown in the examples, is similar to what is shown.  

All RAJA programming model support features are enabled via CMake options,
which are described in :ref:`configopt-label`. 

For the purposes of the discussion for each example, we
assume that any and all data used has been properly allocated and initialized.
This is done in the example code files, but is not discussed further here.

.. toctree::
   :maxdepth: 1

   tutorial/add_vectors.rst
   tutorial/dot_product.rst
   tutorial/indexset_segments.rst
   tutorial/vertexsum_coloring.rst
   tutorial/reductions.rst
   tutorial/atomic_binning.rst
   tutorial/scan.rst
   tutorial/matrix_multiply.rst
   tutorial/permuted-layout.rst
   tutorial/offset-layout.rst
   tutorial/nested_loop_reorder.rst
   tutorial/complex_loop_examples.rst
