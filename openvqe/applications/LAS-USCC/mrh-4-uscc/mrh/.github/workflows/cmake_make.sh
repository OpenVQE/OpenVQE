#!/usr/bin/env bash

set -e

PYSCFLIB=$(python -c "if True:
    import os
    from pyscf import lib
    print (os.path.dirname (lib.__file__))")
cd ./lib
mkdir build ; cd build
cmake -DPYSCFLIB="${PYSCFLIB}" ..
make -j4
cd ..
rm -Rf build
cd ..

