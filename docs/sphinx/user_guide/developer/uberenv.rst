.. developer_guide:

===============
Developer Guide
===============

RAJA shares its Uberenv workflow with other projects. The documentation is
therefore `shared`_.

.. shared: <https://radiuss-ci.readthedocs.io/en/latest/uberenv.html#uberenv-guide)

This page will provides some RAJA specific examples to illustrate the
workflow described in the documentation.

Preliminary considerations
--------------------------

First of all, it is worth noting that RAJA does not have dependencies, except
for CMake, which is most of the time installed externally.

That does not make the workflow useless:
Uberenv will drive Spack which will generate a host-config file with the
toolchain (including gpu support) and the options or variants pre-configured.

Machine specific configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  $ ls -c1 scripts/uberenv/spack_configs
  blueos_3_ppc64le_ib
  darwin
  toss_3_x86_64_ib
  blueos_3_ppc64le_ib_p9
  config.yaml

RAJA has been configured for ``toss_3_x86_64_ib`` and other systems.

Vetted specs
^^^^^^^^^^^^

.. code-block:: bash

  $ ls -c1 .gitlab/*jobs.yml
  .gitlab/lassen-jobs.yml
  .gitlab/quartz-jobs.yml

CI contains jobs for quartz.

.. code-block:: bash

  $ git grep -h "SPEC" .gitlab/quartz-jobs.yml | grep "gcc"
      SPEC: "%gcc@4.9.3"
      SPEC: "%gcc@6.1.0"
      SPEC: "%gcc@7.3.0"
      SPEC: "%gcc@8.1.0"

We now have a list of the specs vetted on ``quartz``/``toss_3_x86_64_ib``.

.. note::
  In practice, one should check if the job is not *allowed to fail*, or even deactivated.

MacOS case
^^^^^^^^^^

In RAJA, the Spack configuration for MacOS contains the default compilers depending on the OS version (`compilers.yaml`), and a commented section to illustrate how to add `CMake` as an external package. You may install CMake with homebrew, for example.


Using Uberenv to generate the host-config file
----------------------------------------------

We have seen that we can safely use `gcc@8.1.0` on quartz. Let us ask for the default configuration first, and then produce static libs, have OpenMP support and run the benchmarks:

.. code-block:: bash

  $ python scripts/uberenv/uberenv.py --spec="%gcc@8.1.0"
  $ python scripts/uberenv/uberenv.py --spec="%gcc@8.1.0~shared+openmp tests=benchmarks"

Each will generate a CMake cache file, e.g.:

.. code-block:: bash

  hc-quartz-toss_3_x86_64_ib-gcc@8.1.0-fjcjwd6ec3uen5rh6msdqujydsj74ubf.cmake

Using host-config files to build RAJA
-------------------------------------

.. code-block:: bash

  $ mkdir build && cd build
  $ cmake -C <path_to>/<host-config>.cmake ..
  $ cmake --build -j .
  $ ctest --output-on-failure -T test

It is also possible to use this configuration with the CI script outside of CI:

.. code-block:: bash

  $ HOST_CONFIG=<path_to>/<host-config>.cmake scripts/gitlab/build_and_test.sh
