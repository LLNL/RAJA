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

To understand the examples that run on a GPU device, it is important to note
that any lambda expression that is defined outside of a GPU kernel and passed
to GPU kernel must decorated with the ``__device__`` attribute when it is 
defined. This can be done directly or by using the ``RAJA_DEVICE`` macro.

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

===============================
A Little C++ Lambda Background
===============================

RAJA is used most easily and effectively by employing C++ lambda expressions,
especially for the bodies of loop kernels. Lambda expressions were 
introduced in C++ 11 to provide a lexically-scoped name binding; i.e., a 
*closure* that stores a function with a data environment. In particular, a 
lambda has the ability to *capture* variables from an enclosing scope for use 
within the local scope of the lambda expression. 

Here, we provide a brief description of the basic elements of C++ lambdas.
A more technical and detailed discussion is available here:
`Lambda Functions in C++11 - the Definitive Guide <https://www.cprogramming.com/c++11/c++11-lambda-closures.html>`_ 

A C++ lambda expression has the following form::

  [capture list] (parameter list) {function body}

The ``capture list`` specifies how variables outside the lambda scope are pulled
into the lambda data environment. The ``parameter list`` defines arguments 
passed to the lambda function body -- for the most part, lambda arguments
are just like arguments for a standard C++ method. RAJA template methods pass 
arguments to lambdas based on usage and context. Values in the capture list 
are initialized when the lambda is created, while values in the parameter list 
are set when the lambda function is called. The body of a lambda function is
similar to the body of an ordinary C++ method.

A C++ lambda can capture values in the capture list by value or by reference.
This is similar to how arguments to C++ methods are passed; e.g., 
pass-by-reference, pass-by-value, etc. However, there are some subtle 
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

.. note:: For most RAJA usage, it is recommended to capture all lambda 
          variables `by-value`. This is required for executing a RAJA loop 
          on a device, such as a GPU via CUDA, and doing so will allow the 
          code to be portable for either CPU or GPU execution.


=========
Examples
=========

The remainder of this tutorial illustrates how to exercise various RAJA 
features using simple examples. Note that all the examples employ
RAJA traversal template methods, which are described briefly 
here :ref:`loop_basic-label`. For the purposes of the discussion, we
assume that any and all data used has been properly allocated and initialized.
This is done in the code examples, but is not discussed further here.

Also, the examples demonstrate CPU execution (sequential, SIMD, openmp 
multi-threading) and CUDA GPU execution only. RAJA also has support for
Intel Threading Building Blocks (TBB) and OpenACC. These features are
enabled with CMake options similar to other programming models. Also, they
are considered experimental; they are described in :ref:`policies-label`
for reference.

.. toctree::
   :maxdepth: 1

   tutorial/add_vectors.rst
   tutorial/dot_product.rst
   tutorial/indexset_segments.rst
   tutorial/vertexsum_coloring.rst
   tutorial/matrix_multiply.rst
   tutorial/nested_loop_reorder.rst
   tutorial/complex_loops-intro.rst
   tutorial/complex_loops-shmem.rst
   tutorial/reductions.rst
   tutorial/atomic_binning.rst
   tutorial/scan.rst
