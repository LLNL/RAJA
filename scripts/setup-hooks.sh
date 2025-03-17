#!/usr/bin/env bash

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
#  and RAJA project contributors. See the RAJA/LICENSE file for details.
#
#  SPDX-License-Identifier: (BSD-3-Clause)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#------------------------------------------------------------------------------
# This script installs client-side hooks
#------------------------------------------------------------------------------
basedir=`git rev-parse --show-toplevel`
hooksdir="$basedir/.git/hooks/"
cp -v $basedir/scripts/githooks/pre-commit $hooksdir
