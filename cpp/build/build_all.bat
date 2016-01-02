@echo off
setlocal

cmake ..
rem cmake -DCMAKE_BUILD_TYPE=Release ..
rem cmake -DCMAKE_BUILD_TYPE=Debug ..

make
make test
make doc

endlocal
echo on
