PREFIX=../INCLUDE/RAJA
PYTHON=/usr/bin/env

$PYTHON ./genForallN.py 2 > $PREFIX/forall2.hxx
$PYTHON ./genForallN.py 3 > $PREFIX/forall3.hxx
$PYTHON ./genForallN.py 4 > $PREFIX/forall4.hxx
$PYTHON ./genForallN.py 5 > $PREFIX/forall5.hxx

$PYTHON ./genLayout.py 5 > $PREFIX/Layout.hxx

$PYTHON ./genView.py 5 > $PREFIX/View.hxx


