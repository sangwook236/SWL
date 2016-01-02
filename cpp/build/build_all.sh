#!/usr/bin/env bash

cmake ..
#cmake -DCMAKE_BUILD_TYPE=Release ..
#cmake -DCMAKE_BUILD_TYPE=Debug ..

make
make test
make doc

#export LD_LIBRARY_PATH=../lib:$LD_LIBRARY_PATH
