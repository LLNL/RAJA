.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _tests-label:

***************************
RAJA Tests
***************************

As noted in :ref:`ci-label`, all RAJA test checks must pass before any PR 
contribution will be merged. Additionally, we recommend that contributors
also include new tests in their code contributions when adding new features
and bug fixes.

.. note:: If RAJA team members think adequate testing is not included in a 
          PR branch, they will ask for additional testing to be added during
          the review process..


=========================
Test Organization
=========================

All RAJA tests are in the ``RAJA/test`` directory and are split into 
*unit tests* and *functional tests*. Unit tests are intended to test basic
interfaces and features of individual classes, methods, etc. Functional tests
are used to test combinations of RAJA features. We have organized our 
tests to make it easy to see what is being tested and easy to add new tests.
For example, tests for each programming model back-end are exercised using
the same common, parameterized test code to ensure back-end support is
consistent.

.. important:: Please follow the sub-directory structure and code implementation
               patterns for existing tests in the ``RAJA/test`` directory when 
               adding or modifying tests. 
