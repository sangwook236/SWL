@echo off
setlocal

rem Tests run in files ending with '_test'.
rem call python -m unittest discover -p "*_test.py"
call python -m unittest discover -p "*_test.py" -v

endlocal
echo on
