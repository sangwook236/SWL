@echo off
setlocal

rem Tests run in files starting with 'test_' or ending with '_test'.
rem call pytest
call python -m pytest

endlocal
echo on
