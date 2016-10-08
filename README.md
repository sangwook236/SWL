# Sang-Wook's Library (SWL)

## Introduction

The goal of SWL is to develop a general-purpose library/framework for a variety of scientific and engineering problems.

Programming languages supported by SWL are as follows:
* C/C++
* C#
* Java
* Matlab

System platforms supported by SWL are as follows:
* Linux/Unix
* Windows
* Mac OS

## Installation

#### Using CMake
* Download sources from the SWL's repository. 
* Change the directory to SWL. 
* Make a build directory. 
	* e.g.) mkdir my_build_dir 
* Change the direcotry to the build directory. 
	* e.g.) cd my_build_dir 
* Run CMake. 
* Configure and generate in CMake. 
* Run make. 
	* make 
	* make test 
	* make doc 

#### Using IDE
SWL supports Code::Blocks and Visual Studio.
* Use Code::Blocks in the Unix-like systems:
	* Use build/*.workspace
* Use Visual Studio in Windows:
	* Use build/*.sln

#### Using a build script 
* Download sources from the SWL's repository. 
* Change the directory to SWL. 
* Change the direcotry to the build directory. 
	* e.g.) cd build 
* Run a build script. 
	* In the Unix-like systems: 
		* ./build_all.sh 
	* In Windows: 
		* build_all.bat 
