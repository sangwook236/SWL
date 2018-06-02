#!/usr/bin/env bash

cmake ..
#cmake -DCMAKE_BUILD_TYPE=Release ..
#cmake -DCMAKE_BUILD_TYPE=Debug ..

make -j4
make -j4 test
make -j4 doc

#export LD_LIBRARY_PATH=../lib:$LD_LIBRARY_PATH
