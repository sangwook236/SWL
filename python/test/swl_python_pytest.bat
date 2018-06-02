@echo off
setlocal

rem Runs tests in files starting with 'test_' or ending with '_test'.
rem pytest
python -m pytest

endlocal
echo on
