.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and other RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _vectorization-label:

==========================
Vectorization (SIMD/SIMT)
==========================

.. warning:: **This section describes an initial draft of an experimental 
             capability that in incomplete and not considered ready
             for production. A basic description is provided here so
             that interested users can take a look, try it out, and 
             provide feedback if they wish to do so.** 

The RAJA team is experimenting with abstractions to provide an API for
SIMD/SIMT programming that *guarantees* that the specified vectorization
occurs without needing to explicitly use intrinsics in user code or 
rely on compiler auto-vectorization implementations.

.. note:: All RAJA vectorization types are in the namespace ``RAJA::expt``.

Currently, the key abstractions available in RAJA are:

  * ``Register``, which wraps the underlying SIMD/SIMT registers and provides 
    consistent uniform access to underlying hardware, using intrinsics
    when possible. The RAJA register abstraction currently supports: avx, 
    avx2, avx512, CUDA, and HIP enabled hardware and compilers.
  * ``Vector``, which builds on ``Register`` to provide arbitrary length
    vectors and operations on them.
  * ``Matrix``, which builds on ``Register`` to provide arbitrary-sized
    matrices, column-major and row-major layouts, and operations on them.

Finally, these capabilities integrate with the RAJA :ref:`view-label` 
capabilities, which implements am expression-template system that allows 
a user to write linear algebra expressions on arbitrarily sized scalars, 
vectors, and matrices and have the appropriate SIMD/SIMT instructions
performed during expression evaluation.


----------------------
A Motivating Example
----------------------

Fill this in if we want to start with something...

------------------------
Why Are We Doing This?
------------------------

Quoting Tim Foley in `Matt Pharr's blog <https://pharr.org/matt/blog/2018/04/18/ispc-origins>`_: "Auto-vectorization is not a programming model". Unless, of
course, you consider "hope for the best" to be a sound plan.

Auto-vectorization is problematic for multiple reasons. With auto-vectorization,
vectorization is not explicit in the source code, so a compiler must divine 
correctness when attempting to vectorize. Since most compilers are very 
conservative in this regard, many vectorization opportunities will most
likely be missed. Also, every compiler will treat your code differently since
implementations use different heuristics, even for different versions of the 
same compiler. In general, it is impossible for most application developers
to clearly understand the decisions made by a compiler during its optimization
process. 

Using vectorization intrinsics in application source code is also problematic 
because different processors support different instruction set architectures
(ISAs) and so source code portability requires a mechanism that insulates it 
from architecture-specific calls.

GPU programming makes us be explicit about parallelization, and SIMD 
is really no different. RAJA enables single-source portable code across a 
variety of programming model back-ends. The RAJA vectorization abstractions
introduced here are an attempt to bring a level of convergence between SIMD 
and GPU programming by providing uniform access to hardware-specific 
acceleration.

.. note:: **Auto-vectorization is not a programming model.** --Tim Foley

---------------------
Register
---------------------

``RAJA::expt::Register<T, REGISTER_POLICY>`` is a class template that takes a
a data type parameter ``T`` and a register policy ``REGISTER_POLICY`` that
indicates the hardware register type. The ``RAJA::expt::Register`` provides 
uniform access to register-level operations and is intended as a building
block for higher level abstractions. A ``RAJA::expt::Register`` type represents
one SIMD register for a CPU architecture and 1 value/lane for a GPU 
architecture. 

.. note:: A user can code directly to the ``RAJA::expt::Register`` type, but we
          don't recommend this. Instead, we recommend use the higher level
          abstractions RAJA provides.

``RAJA::expt::Register`` supports four scalar element types, ``int32_t``, 
``int64_t``, ``float``, and ``double``. These are the only types that are 
portable across all SIMD/SIMT architectures. ``Bfloat``, for example, is not 
portable, so we don't provide support for that type.

``RAJA::expt::Register`` supports various SIMD/SIMT hardware implementations, 
including: SIMD (CPU vectorization) AVX, AVX2, and AVX512, CUDA warp, and
HIP wavefront. Scalar support is provided for all hardware. Extensions to 
support other architectures should be straightforward.

Register Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``RAJA::expt::Register`` provides various operations, including:

  * Basic SIMD handling: get element, broadcast
  * Memory operations: load (packed, strided, gather) and store (packed, strided, scatter)
  * SIMD element-wise arithmetic: add, subtract, multiply, divide, vmin, vmax
  * Reductions: dot-product, sum, min, max
  * Special operations for matrix operations: permutations, segmented operations

.. note: All operations are provided for all hardware. Depending on hardware
         support, some operations may have slower serial performance; 
         e.g., gather/scatter.

Register DAXPY Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following is a code example that shows using the ``RAJA::expt::Register`` 
class to perform a *DAXPY* kernel with 'avx2' CPU SIMD instructions::

  // define array length
  int len = ...;

  // data used in kernel
  double a = ...;
  double const *X = ...; 
  double const *Y = ...; 
  double *Z = ...; 

  using reg_t = RAJA::expt::Register<double, RAJA::expt::avx2_register>;
  int reg_width = reg_t::s_num_elem;    // width of avx2 register is 4 doubles	

  // Compute daxpy in chunks of 4 values at one time
  for (int i = 0;i < len; i += reg_width){
    reg_t x, y;
    
    // load 4 consecutive values of X, Y arrays into registers
    x.load_packed( X+i );
    y.load_packed( Y+i );

    // perform daxy on 4 values simultaneously (store in register)
    reg_t z = a * x + y;

    // store register result in Z array
    z.store_packed( Z+i );
  }

  // loop 'postamble' code.
  int remainder = len % reg_width;
  if (remainder) {
    reg_t x, y;

    // 'i' is the starting array index of the remainder
    int i = len - remainder;
       
    // load remainder values of X, Y arrays into registers 
    x.load_packed_n( X+i, remainder );
    y.load_packed_n( Y+i, remainder );

    // perform daxy on remainder values simultaneously (store in register)
    reg_t z = a * x + y;

    // store register result in Z array
    z.store_packed_n(Z+i, remainder);
  }

The code is guaranteed to vectorize since the ``RAJA::expt::Register`` 
operations insert the proper SIMD instructions into the method calls. Note 
that the ``RAJA::expt::Register`` provides overloads of basic arithmetic 
operations so that the DAXPY operation itself (z = a * x + y) looks like 
vanilla scalar code.

However, since we are using bare pointers to the data, the load and store 
operations are explicit. Also, we have to write the (duplicate) postamble 
code to handle the case where the array length (len) is not an integer 
multiple of the register width to perform the DAXPY operation on the 
*remainder* of the array that remains after the for-loop.

These extra lines of code should make it clear why we do not recommend
using ``RAJA::Register`` directly in application code.


-------------------
Tensor Register
-------------------

``RAJA::expt::TensorRegister< >`` is a class template that provides a 
higher-level, user-facing interface on top of the ``RAJA::expt::Register`` 
class.  ``RAJA::expt::TensorRegister< >`` wraps one or more 
``RAJA::expt::Register< >`` objects to create a tensor-like object.

.. note:: As with ``RAJA::expt::Register``, we don't recommend using 
          ``RAJA::expt::TensorRegister`` directly. Rather use more-specific 
          type aliases that RAJA provides and which are described below.

**To make code cleaner and more readable, the specific types are intended to
be used with ``RAJA::View`` and ``RAJA::expt::TensorIndex`` objects.

Vector Register
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``RAJA::expt::VectorRegister<T, REGISTER_POLICY, NUM_ELEM>`` provides an 
abstraction for a vector of arbitrary length. It is implemented using one or 
more ``RAJA::expt::Register`` objects. The vector length is independent of the 
underlying register width. The template parameters are: ``T`` data type, 
``REGISTER_POLICY`` vector register policy, and ``NUM_ELEM`` number of 
data elements of type ``T`` that fit in a register. The last two of these
have defaults for all cases, so they do not usually need to be specified.

Earlier we stated that it is not recommended to use ``RAJA::expt::Register``
directly. The reason for this is that it is good to decouple
vector length from hardware register size since it allows one to write
simpler, more readable code that is easier to get correct. This should be 
clear from the code example below.

Vector Register DAXPY Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following code example shows the *DAXPY* computation shown above written 
using ``RAJA::expt::VectorRegister``, ``RAJA::expt::VectorIndex``, and 
``RAJA::View`` classes, which obviate the need for the extra lines of code 
discussed earlier::

  // define array length and data used in kernel (as before)
  int len = ...;
  double a = ...;
  double const *X = ...;
  double const *Y = ...;
  double *Z = ...;

  // define vector register and index types
  using vec_t = RAJA::expt::VectorRegister<double, RAJA::expt::avx2_register>;
  using idx_t = RAJA::expt::VectorIndex<int, vec_t>;

  // wrap array pointers in RAJA View objects   
  auto vX = RAJA::make_view( X, len );
  auto vY = RAJA::make_view( Y, len );
  auto vZ = RAJA::make_view( Z, len );

  //  'all' knows the SIMD chunk size based on the register type
  auto all = idx_t::all();

  // compute the complete array daxpy in one line of code
  vZ( all ) = a * vX( all ) + vY( all );

This code has several advantages over the previous example. It is guaranteed 
to vectorize and is much easier to read, get correct, and maintain since 
the ``RAJA::View`` class handles the looping and postamble code automatically 
to allow arrays of arbitrary size. The ``RAJA::View`` class provides overloads 
of the arithmetic operations based on the ``all`` type and inserts the 
appropriate SIMD instructions and load/store operations to vectorize the 
operations as in the earlier example. It may be considered by some to be 
inconvenient to have to use the ``RAJA::View`` class, but it is easy to wrap 
bare pointers as shown in the example.

CPU/GPU Portability
^^^^^^^^^^^^^^^^^^^^^

It is important to note that the code example in the previous section is 
*not* portable to run on a GPU in this form because it does not include a 
way to launch a GPU kernel. The following code example shows how to enable the 
code to run on either a CPU or GPU via a run time choice::

  // array lengths and data used in kernel same as above

  // define vector register and index types
  using vec_t = RAJA::expt::VectorRegister<double>;
  using idx_t = RAJA::expt::VectorIndex<int, vec_t>;

  // array pointers wrapped in RAJA View objects as before

  using cpu_launch = RAJA::expt::seq_launch_t;
  using gpu_launch = RAJA::expt::cuda_launch_t<false>; // false => launch
                                                       // CUDA kernel
                                                       // synchronously

  using pol_t = 
    RAJA::expt::LoopPolicy< cpu_launch, gpu_launch >;

  RAJA::expt::ExecPlace cpu_or_gpu = ...;

  RAJA::expt::launch<pol_t>( cpu_or_gpu, resources,

                             [=] RAJA_HOST_DEVICE (context ctx) {
                                 auto all = idx_t::all();
                                 vZ( all ) = a * vX( all ) + vY( all );
                             }
                           );

This version of the kernel will run on a CPU or GPU depending on the run time
chosen value of the variable ``cpu_or_gpu``. When compiled, the code will 
generate versions of the kernel for the CPU and GPU based on the parameters 
in the ``pol_t`` loop policy. The CPU version will be the same as the version
in the previous section. The GPU version will be essentially the same but
run in a GPU kernel. Note that there is only one template argument passed to 
the register when ``vec_t`` is defined. ``RAJA::expt::VectorRegister<double>``
uses defaults for the register policy, based on the system hardware, and 
number of data elements of type double that will fit in a register.

Matrix Registers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RAJA provides ``RAJA::expt::TensorRegister`` type aliases to support
matrices of arbitrary size and shape. These are:

  * ``RAJA::expt::SquaretMatrixRegister<T, LAYOUT, REGISTER_POLICY>`` which
    abstracts an N x N square matrix.
  * ``RAJA::expt::RectMatrixRegister<T, LAYOUT, ROWS, COLS, REGISTER_POLICY>`` 
     which abstracts an N x M rectangular matrix.

The matrices are implemented using one or more ``RAJA::expt::Register`` 
objects. Data layout can be row-major or column major. Matrices are intended 
to be used with ``RAJA::View`` and ``RAJA::expt::TensorIndex`` objects,
similar to what was shown above with ``RAJA::expt::VectorRegister``

Matrix operations support matrix-matrix, matrix-vector, vector-matrix 
multiplication, and transpose operations. Rows or columns can be represented
with one or more registers, or a power-of-two fraction of a single register.
This is important for CUDA GPU warp registers, which are 32-wide, and HIP
GPU wavefront registers, which are 64-wide.

Here is a simple code example that performs the matrix-analogue of the 
vector DAXPY operation presented above using square matrices::

  // define matrix size and data used in kernel (similar to before)
  int N = ...;
  double a = ...;
  double const *X = ...;
  double const *Y = ...;
  double *Z = ...;

  // define matrix register and row/column index types
  using mat_t = RAJA::expt::SquareMatrixRegister<double, 
                                                 RAJA::expt::RowMajorLayout>;
  using row_t = RAJA::expt::RowIndex<int, mat_t>;
  using col_t = RAJA::expt::ColIndex<int, mat_t>;

  // wrap array pointers in RAJA View objects (similar to before)
  auto mX = RAJA::make_view( X, N, N );
  auto mY = RAJA::make_view( Y, N, N );
  auto mZ = RAJA::make_view( Z, N, N );

  using cpu_launch = RAJA::expt::seq_launch_t;
  using gpu_launch = RAJA::expt::cuda_launch_t<false>; // false => launch
                                                       // CUDA kernel
                                                       // synchronously
  using pol_t =
    RAJA::expt::LoopPolicy< cpu_launch, gpu_launch >;

  RAJA::expt::ExecPlace cpu_or_gpu = ...;

  RAJA::expt::launch<pol_t>( cpu_or_gpu, resources,

      [=] RAJA_HOST_DEVICE (context ctx) {
         auto rows = row_t::all();
         auto cols = col_t::all();
         mZ( rows, cols ) = a * mX( rows, colsall ) + mY( rows, colsall );
      }
    ); 

Conceptually, as well as implementation-wise, this is similar to the previous
vector example except the operations are in two dimensions. The kernel code is 
easy to read, it is guaranteed to vectorize, and iterating over the data is 
handled by RAJA (register width sized chunk, plus postamble scalar operations).
Again, the ``RAJA::View`` arithmetic operation overloads insert the 
appropriate vector instructions in the code.


------------------------------------
RAJA Views and Expression Templates
------------------------------------

Include discussion and figure from Adam's presentation...
