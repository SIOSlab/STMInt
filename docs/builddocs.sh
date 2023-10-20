#!/bin/bash
# Script to fully regenerate, compile and update html docs
# must be run directly from documentation directory

if [ ! -d "../STMint" ] || [ `basename $PWD` != "docs" ] ; then
    echo "This script must be run from the docs directory in the STMint parent directory."
    exit 1
fi


sphinx-apidoc -f -o . ../STMint/

rm modules.rst

make html
make html

exit 0