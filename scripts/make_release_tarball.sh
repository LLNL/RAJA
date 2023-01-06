#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

TAR_CMD=`which tar`
VERSION=`git describe --tags`

git archive --prefix=RAJA-${VERSION}/ -o RAJA-${VERSION}.tar HEAD 2> /dev/null

echo "Running git archive submodules..."

p=`pwd` && (echo .; git submodule foreach --recursive) | while read entering path; do
    temp="${path%\'}";
    temp="${temp#\'}";
    path=$temp;
    [ "$path" = "" ] && continue;
    (cd $path && git archive --prefix=RAJA-${VERSION}/$path/ HEAD > $p/tmp.tar && ${TAR_CMD} --concatenate --file=$p/RAJA-${VERSION}.tar $p/tmp.tar && rm $p/tmp.tar);
done

gzip RAJA-${VERSION}.tar
