#!/usr/bin/env zsh

if (( $# == 0 )) ; then
  dockerfiles=(**/Dockerfile)
else
  dockerfiles=($@)
fi

all_images="${compiler_images} ubuntu-clang-base"

function build-tag () {
  echo building $2 and tagging with $1
  docker build --tag $1 $2
}

build-tag rajaorg/compiler:ubuntu-clang-base ubuntu-clang-base

for df in ${dockerfiles} ; do
  imgname=${df:h}
  imgpath=rajaorg/compiler:$imgname
  [[ ${imgname} == 'ubuntu-clang-base' ]] && continue
  build-tag $imgpath $imgname

  echo pushing $imgpath
  docker push $imgpath
done

for img in $all_images ; do
  # docker push rajaorg/compiler:$img
done 
