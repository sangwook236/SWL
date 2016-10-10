# Sang-Wook's Library for C++ (SWL-C++)

## Introduction

## Building Library

### Using CMake
* Change the directory to SWL-C++.
* Make a build directory.
	* e.g.) mkdir my_build_dir
* Change the directory to the build directory.
	* e.g.) cd my_build_dir
* Run [CMake](https://cmake.org/documentation/).
* Configure and generate in CMake.
* Run make.
	* make
	* make test
	* make doc

### Using IDE
SWL supports [Code::Blocks](http://www.codeblocks.org/) and [Visual Studio](https://www.visualstudio.com/).
* Change the directory to SWL-C++.
* Run IDE.
	* Use Code::Blocks in the Unix-like systems.
	* Use Visual Studio in Windows.
* Open a file.
	* Open build/*.workspace in Code::Blocks.
	* Open build/*.sln in Visual Studio.
* Build.

### Using a build script
* Change the directory to SWL-C++.
* Change the directory to build.
	* cd build
* Run a build script.
	* In the Unix-like systems:
		* ./build_all.sh
	* In Windows:
		* build_all.bat

## Document
Use the SWL's doxygen configuation file, doc/swl.doxy.
* Change the directory to SWL-C++.
* Change the directory to doc.
	* cd doc
* Run [Doxygen](https://www.stack.nl/~dimitri/doxygen/manual/) command.
	* doxygen swl.doxy
* Open an HTML page or a RTF file.
	* Open doc/html/index.html.
	* Open doc/rtf/refman.rtf.
