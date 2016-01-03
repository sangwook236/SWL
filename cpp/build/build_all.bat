@echo off
setlocal

cmake ..
rem cmake -DCMAKE_BUILD_TYPE=Release ..
rem cmake -DCMAKE_BUILD_TYPE=Debug ..

make -j4
make -j4 test
make -j4 doc

endlocal
echo on
