# Sang-Wook's Library for C++ (SWL-C++)

## Building Library

### Using CMake
* Change the directory to SWL.
* Make a build directory.
	* e.g.) mkdir my_build_dir
* Change the directory to the build directory.
	* e.g.) cd my_build_dir
* Run CMake.
* Configure and generate in CMake.
* Run make.
	* make
	* make test
	* make doc

### Using IDE
SWL supports Code::Blocks and Visual Studio.
* Change the directory to SWL.
* Open IDE.
	* Use Code::Blocks in the Unix-like systems:
		* build/*.workspace
	* Use Visual Studio in Windows:
		* build/*.sln

### Using a build script
* Change the directory to SWL.
* Change the directory to build.
	* cd build
* Run a build script.
	* In the Unix-like systems:
		* ./build_all.sh
	* In Windows:
		* build_all.bat
