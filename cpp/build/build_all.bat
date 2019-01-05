@echo off
setlocal

call cmake ..
rem call cmake -DCMAKE_BUILD_TYPE=Release ..
rem call cmake -DCMAKE_BUILD_TYPE=Debug ..

call make -j4
call make -j4 test
call make -j4 doc

endlocal
echo on
