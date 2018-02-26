compiler_images="gcc49 gcc5 gcc6 gcc7 clang5 clang4"
all_images="${compiler_images} ubuntu-clang-base"

for img in $all_images ; do
  docker push trws/raja-testing:$img
done

# for img in ${compiler_images} ; do
#   echo docker build --tag trws/raja-testing:$img $img
#   docker build --tag trws/raja-testing:$img $img
# done

