#!/bin/bash

TAR_CMD=gtar
VERSION=0.6.0rc1

git archive --prefix=RAJA-${VERSION}/ -o RAJA-${VERSION}.tar HEAD 2> /dev/null

echo "Running git archive submodules..."

p=`pwd` && (echo .; git submodule foreach) | while read entering path; do
    temp="${path%\'}";
    temp="${temp#\'}";
    path=$temp;
    [ "$path" = "" ] && continue;
    (cd $path && git archive --prefix=RAJA-${VERSION}/$path/ HEAD > $p/tmp.tar && ${TAR_CMD} --concatenate --file=$p/RAJA-${VERSION}.tar $p/tmp.tar && rm $p/tmp.tar);
done

gzip RAJA-${VERSION}.tar
