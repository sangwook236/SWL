@echo off
setlocal

cmake ..
make
make test
make doc

endlocal
echo on
