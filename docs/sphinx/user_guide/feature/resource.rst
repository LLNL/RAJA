.. ##
.. ## Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
.. ## and other RAJA project contributors. See the RAJA/COPYRIGHT file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _resource-label:

=========
Resources
=========

This section describes the basic concepts of Resource types and their functionality in ``RAJA::forall``. Resources are used as an interface to various backend constructs for their respective hardware. Currently there exists Resource types for ``Cuda``, ``Hip``, ``OpenMP-target`` and ``Host``. These resources are designed to allow the user to execute ``RAJA::forall`` calls asynchronously on a respective thread/stream. The underlying concept of each individual Resource is still under development and it should be considered that functionality / behaivour may change.

.. note:: * Currently feature complete asynchronous behaviour and streamed/threaded support is
            only available when using ``Cuda`` or ``Hip`` resources. 
          * The ``RAJA::resources`` namespace aliases the ``camp::resources`` namespace.

Each resource has a set of underlying functionality that is default across all resources.  

 ===================== ===============================================
 Methods               Brief description
 ===================== ===============================================
 get_platform          Returns the underlying camp platform
                       the resource is associated with.
 get_event             Return an Event object for the resource from
                       the last resource call.
 allocate              Allocate data per the resources given
                       backend.
 deallocate            Deallocate data per the resources given
                       backend.
 memcpy                Perform a memory copy from a src location
                       to a destination location from the
                       resources backend.
 memset                Set memory value per the resourses
                       given backend.
 wait_for              Enqueuea wait on the resources stream/thread
                       for a user passed event to occur.
 ===================== ===============================================
  
Each resource type also defines specific backend information/functionality. For example, each
Cuda resource contains a ``cudaStream_t`` value with an associated get method. See the 
individual functionality for each resource in ``raja/tpl/camp/include/resource/``.


------------
Type-Erasure
------------

Resources can be declared in two formats: An erased resource, and a concrete resource. When 
declared on the same backend, their underlying runtime functionality is the same. The 
erased resource allows a user the ability to change a resourse backend during runtime. 

Concrete Cuda resource::

    RAJA::resources::Cuda my_cuda_res;

Erased Cuda resource::

    RAJA::resources::Resource my_res{RAJA::resources::Cuda()};

Memory allocation on Cuda resources::

    int* a1 = my_cuda_res.allocate<int>(ARRAY_SIZE);
    int* a2 = my_res.allocate<int>(ARRAY_SIZE);

In this case both a1 and a2 are allocated on the GPU since the underlying resource of ``my_res`` is a
Cuda resource.


------
Forall
------

A resource is an optional argument to a ``RAJA::forall`` call. It is passed as the first argument 
when the forall is templated on the Execution policy, or the second argument when the execution
policy is passed by value as such::

    RAJA::forall<ExecPol>(my_cuda_res, .... )

Or::

    RAJA::forall(ExecPol(), my_cuda_res, .... )

This maintains the order of conditions that can be passed to a ``RAJA::forall`` call.

When specifying a Cuda/Hip resource the RAJA kernel is executed aynchronously on a stream.
Currently Cuda/Hip are the only Resources that enable asynchronous threading with a ``RAJA::forall``.
Currently all other calls are defaulted to using the ``Host`` resource until further support is 
added.

The Resource type that is passed to a ``RAJA::forall`` call must be a concrete type. This is to
allow compile time assertion that the resource is compatible with the given execution policy. For
example::
    
    using ExecPol = RAJA::cuda_exec_async<BLOCK_SIZE>;
    RAJA::resources::Cuda my_cuda_res;
    RAJA::resources::Resource my_res{RAJA::resources::Cuda()};

    RAJA::forall<ExecPol>(my_cuda_res, .... ) // Compiles.
    RAJA::forall<ExecPol>(my_res, .... ) // Compilation Error.

Below is a list of the current concrete resource type execution policy suport.

 ======== ==============================
 Resource Policy Type
 ======== ==============================
 Cuda     cuda_exec

          cuda_exec_async

 Hip      hip_exec

          hip_exec_async

 Omp*     omp_target_parallel_for_exec
          omp_target_parallel_for_exec_n
 Host     loop_exec
          seq_exec
          openmp_parallel_exec
          omp_for_schedule_exec
          omp_for_nowait_schedule_exec
          simd_exec
          tbb_for_dynamic
          tbb_for_static
 ======== ==============================

.. note:: * The ``RAJA::resources::Omp`` resource still under development.

IndexSet policies require two execution policies to define them. Currently a users only need to pass a
single resource to the forall Indexset call. This resource will be used to execute the inner 
execution loop of the indexset policy.::

    using ExecPol = RAJA::ExecPolicy<RAJA::seq_segit, RAJA::cuda_exec<256>>;
    RAJA::forall<ExecPol>(my_cuda_res, iset,  .... );

When a resource is not provided by the user a *default* resource is assigned from within the RAJA
forall implementation. This default resource can be accessed in a number of ways.

Directly from the concrete resource type::

    RAJA::resources::Cuda my_default_cuda = RAJA::resources::Cuda::get_defualt();

The Resource type can be deduced from an execution policy::

    using Res = RAJA::resources::get_resource<ExecPol>::type;
    Res r = Res::get_defualt();

Deduced from an execution policy and return the default directly::

    auto my_resource = RAJA::resources::get_default_resource<ExecPol>();

.. note:: * For Cuda and Hip the default resource is *NOT* the CUDA or HIP default stream it is it's 
            own stream defined in the ``camp/include/resource/``. This is in an attempt to break away
            from some of the issues that arise from the synchronization behaviour of the CUDA/HIP 
            default stream. It is still possible to use the CUDA/HIP defined default stream as the
            default resource. This can be enabled by defining ``CAMP_USE_PLATFORM_DEFAULT_STREAM``

------
Events
------

Event objects are a feature that allow users to wait or query the status of a Resources action. An event can be returned from a resource with::

    RAJA::resources::Event e = my_res.get_event();

Getting an event like this enqueues an event type object for the given backend. 

You can call a blocking function and wait for that event::

    e.wait();

Preferably users can enqueue the event to a specific resource, forcing that resources to wait for the event::

    my_res.wait_for(&e);

The latter is useful as it allows the user to set up dependencies between resource objects and ``RAJA::forall`` calls.

.. note:: *An Event object is only generated if a user specifically returns one from a ``RAJA::forall``::
           call. This stops unnecessary event object being created and causing a performance hit when not
           needed. For example::
    
               forall<cuda_exec_async<BLOCK_SIZE>>(my_cuda_res, ...

           Will *not* generate a cudaStreamEvent.::

                RAJA::resources::Event e = forall<cuda_exec_async<BLOCK_SIZE>>(my_cuda_res, ...

           Will generate a cudaStreamEvent.

-------
Example
-------

An example of how to use events is shown below. This example executes three kernels accross two cuda streams on the GPU with a dependence on that both the first and second kernel finish execution before the third can begin. It also demonstrates copying memory from the device to host with a resource.
    
First define two concrete CUDA resources and a Host resource::

    RAJA::resources::Cuda dev1;
    RAJA::resources::Cuda dev2;
    RAJA::resources::Host host;

Allocate data on 2 GPU arrays and a host array::

    int* d_array1 = dev1.allocate<int>(ARRAY_SIZE);
    int* d_array2 = dev2.allocate<int>(ARRAY_SIZE);
    int* h_array  = host.allocate<int>(ARRAY_SIZE);

Execute Cuda stream 1::

    forall<EXEC_POLICY>(dev1, RangeSegment(0,ARRAY_SIZE),
      [=] RAJA_HOST_DEVICE (int i) {
        d_array1[i] = i;
      }
    );
    
Execute Cuda stream 2 and return an ``Event`` object::

    resources::Event e = forall<EXEC_POLICY>(dev2, RangeSegment(0,ARRAY_SIZE),
      [=] RAJA_HOST_DEVICE (int i) {
        d_array2[i] = -1;
      }
    );
    
The next kernel on stream 1 requires that the last forall on dev2 finish first so we enqueue a wait to dev1 depending on dev2 finishing::

    dev1.wait_for(&e);
    
Execute the second kernel on stream 1 now that work has finished on the previous two kernels::

    forall<EXEC_POLICY>(dev1, RangeSegment(0,ARRAY_SIZE),
      [=] RAJA_HOST_DEVICE (int i) {
        d_array1[i] *= d_array2[i];
      }
    );
    
We enqueu a memcpy on stream 1 from the GPU to the host.::

    dev1.memcpy(h_array, d_array1, sizeof(int) * ARRAY_SIZE);
    
Finally use the data on the host side.::

    forall<policy::sequential::seq_exec>(host, RangeSegment(0,ARRAY_SIZE),
      [=] (int i) {
        ASSERT_EQ(h_array[i], -i); 
      }
    );
