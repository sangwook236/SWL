project(SWL_CPP)
cmake_minimum_required(VERSION 2.8)

##
## Add path for custom modules.
##
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

##
## Set package information.
##
set(PACKAGE_NAME "Sang-Wook's Library for C++")
set(PACKAGE_VERSION "1.0.0")
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(PACKAGE_WEBSITE "http://www.sangwook.com/")
set(PACKAGE_BUGREPORT "sangwook236@gmail.com")

##
## Set options.
##
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_DOCS "Build documents" ON)

#option(USE_PTHREADS "Thread-safe build (using pthreads)" OFF)
set(USE_PTHREADS OFF)
#option(USE_LAPACK "" ON)
set(USE_LAPACK ON)
#option(USE_BOOST "" ON)
set(USE_BOOST ON)
#option(USE_GSL "" OFF)
set(USE_GSL OFF)
#option(USE_FFTW3 "" OFF)
set(USE_FFTW3 OFF)
#option(USE_OPENCV "" OFF)
set(USE_OPENCV OFF)
#option(USE_PCL "" OFF)
set(USE_PCL OFF)

#option(USE_DOXYGEN "" ON)
set(USE_DOXYGEN ON)
#option(USE_LYX "" OFF)
set(USE_LYX OFF)

if(BUILD_SHARED_LIBS)
	set(LIB_TYPE SHARED)
else(BUILD_SHARED_LIBS)
	set(LIB_TYPE STATIC)
endif(BUILD_SHARED_LIBS)

if(BUILD_TESTS)
	enable_testing()
endif(BUILD_TESTS)

##
## Find packages.
##
if(USE_PTHREADS)
	find_package(pthreads REQUIRED)
	include_directories(${PTHREADS_INCLUDE_DIR})
	add_definitions(-D_REENTRANT)
else(USE_PTHREADS)
	## TODO [check] >>
	find_package(Threads REQUIRED)
	# variables : CMAKE_THREAD_LIBS_INIT, CMAKE_USE_SPROC_INIT, CMAKE_USE_WIN32_THREADS_INIT, CMAKE_USE_PTHREADS_INIT, CMAKE_HP_PTHREADS_INIT, CMAKE_THREAD_PREFER_PTHREAD
endif(USE_PTHREADS)
	
if(USE_LAPACK)
	if(APPLE)
		find_package(vecLib REQUIRED)
#		find_package(LAPACK REQUIRED)
		# variables : LAPACKLIBS_VECLIB_FRAMEWORK, LAPACKLIBS_ACCELERATE_FRAMEWORK, LAPACK_LIBRARIES, BLAS_LIBRARIES

		if(NOT LAPACK_LIBRARIES_DIR)
			set(ATLAS_LIBRARIES_DIR /opt/local/lib)
		else()
			set(ATLAS_LIBRARIES_DIR ${LAPACK_LIBRARIES_DIR})
		endif()
		if(NOT LAPACK_INCLUDE_DIR)
			set(ATLAS_INCLUDE_DIR /opt/local/include)
		else()
			set(ATLAS_INCLUDE_DIR ${LAPACK_LIBRARIES_DIR})
		endif()
	else(APPLE)
		find_package(LAPACK REQUIRED)
		# variables : LAPACK_FOUND, LAPACK_LINKER_FLAGS, LAPACK_LIBRARIES, LAPACK95_LIBRARIES, LAPACK95_FOUND, BLA_STATIC, BLA_VENDOR, BLA_F95

		if(NOT LAPACK_LIBRARIES_DIR)
			set(ATLAS_LIBRARIES_DIR /usr/lib/atlas-base)  # for Ubuntu.
#			set(ATLAS_LIBRARIES_DIR /usr/lib64/atlas)  # for RHEL.
		else()
			set(ATLAS_LIBRARIES_DIR ${LAPACK_LIBRARIES_DIR})
		endif()
		if(NOT LAPACK_INCLUDE_DIR)
			set(ATLAS_INCLUDE_DIR /usr/include/atlas)  # for Ubuntu.
#			set(ATLAS_INCLUDE_DIR /usr/include)  # for RHEL.
		else()
			set(ATLAS_INCLUDE_DIR ${LAPACK_LIBRARIES_DIR})
		endif()
	endif(APPLE)
endif(USE_LAPACK)

if(USE_BOOST)
	find_package(Boost 1.52 REQUIRED)
	include_directories(${Boost_INCLUDE_DIRS})
	link_directories(${Boost_LIBRARY_DIRS})
	# variables : Boost_FOUND, Boost_LIBRARIES, Boost_<C>_FOUND, Boost_<C>_LIBRARY, Boost_VERSION, BOOST_ROOT
endif(USE_BOOST)
if(USE_GSL)
	find_package(GSL REQUIRED)
	include_directories(${GSL_INCLUDE_DIRS})
	# variables : GSL_FOUND, GSL_LIBRARIES, GSL_VERSION, GSL_ROOT_DIR
endif(USE_GSL)

if(USE_FFTW3)
	find_package(FFTW3 REQUIRED)  # error : the file, 'FindFFTW3.cmake' doesn't exist in CMAKE_MODULE_PATH ...
#	set(FFTW3_FOUND NO)  # if FFTW3 is not found or does not try being found, FFTW3_FOUND is set to NO.
endif(USE_FFTW3)
if(USE_OPENCV)
	find_package(OpenCV REQUIRED)
endif(USE_OPENCV)
if(USE_PCL)
	find_package(PCL 1.5 REQUIRED)
endif(USE_PCL)

if(USE_DOXYGEN)
	find_package(Doxygen)
endif(USE_DOXYGEN)
if(USE_LYX)
	find_package(LyX)  # error : the file, 'FindLyX.cmake' doesn't exist in CMAKE_MODULE_PATH ...
#	set(LYX_FOUND NO)  # if LyX is not found or does not try being found, LYX_FOUND is set to NO.
	set(LYX_EXECUTABLE lyx)
endif(USE_LYX)

##
## Set platform specific options.
##	CMAKE_SYSTEM_NAME, CMAKE_SYSTEM
##
if(APPLE)
	message(STATUS ">>---> Platform : Apple <---<<")

	if(POLICY CMP0042)
		cmake_policy(SET CMP0042 NEW)
	endif()

	link_directories(/opt/local/lib)

	if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")  # for Mac OS X.
	endif()
elseif(CYGWIN)
	message(STATUS ">>---> Platform : Cygwin <---<<")
elseif(MSYS)
	message(STATUS ">>---> Platform : MSYS <---<<")
elseif(UNIX AND NOT APPLE AND NOT CYGWIN)
	message(STATUS ">>---> Platform : Unix <---<<")

	if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")  # for Linux.
	endif()
elseif(WIN32 AND NOT CYGWIN)
	message(STATUS ">>---> Platform : WIN32 <---<<")
elseif(MINGW)
	message(STATUS ">>---> Platform : MinGW <---<<")
elseif(BORLAND)
	message(STATUS ">>---> Platform : Borland <---<<")
elseif(WATCOM)
	message(STATUS ">>---> Platform : Watcom <---<<")
elseif(MSVC)
	message(STATUS ">>---> Platform : MSVC <---<<")
	
#	add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
endif()

##
## Set compiler specific options.
##
if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
	message(STATUS ">>---> Compiler : Clang <---<<")

	## Explicitly specify standard c++ library.
#	set(CLANG_LINK_FLAGS "-stdlib=libstdc++")

	## add_definitions() is needed to remove unknown warnings.
	add_definitions(${CLANG_LINK_FLAGS})
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CLANG_LINK_FLAGS}")
elseif(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	message(STATUS ">>---> Compiler : GNU <---<<")
elseif(CMAKE_CXX_COMPILER_ID MATCHES Intel)
	message(STATUS ">>---> Compiler : Intel <---<<")
elseif(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
	message(STATUS ">>---> Compiler : MSVC <---<<")

	set(CMAKE_MFC_FLAG 2)  # Use MFC in a Shared DLL.
#	set(CMAKE_C_FLAGS_DEBUG_INIT "/D_DEBUG /MDd /Zi /Ob0 /Od /RTC1")
#	set(CMAKE_C_FLAGS_MINSIZEREL_INIT "/MD /O1 /Ob1 /D NDEBUG")
#	set(CMAKE_C_FLAGS_RELEASE_INIT "/MD /O2 /Ob2 /D NDEBUG")
#	set(CMAKE_C_FLAGS_RELWITHDEBINFO_INIT "/MD /Zi /O2 /Ob1 /D NDEBUG")
#	set(CMAKE_CXX_FLAGS_DEBUG_INIT "/D_DEBUG /MDd /Zi /Ob0 /Od /RTC1")
#	set(CMAKE_CXX_FLAGS_MINSIZEREL_INIT "/MD /O1 /Ob1 /D NDEBUG")
#	set(CMAKE_CXX_FLAGS_RELEASE_INIT "/MD /O2 /Ob2 /D NDEBUG")
#	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "/MD /Zi /O2 /Ob1 /D NDEBUG")
endif()

##
## Set binary directories.
##
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	message(STATUS ">>---> Build : Debug <---<<")
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/Debug)
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/Debug)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/Debug)
#elseif(CMAKE_BUILD_TYPE MATCHES RELEASE)
else()
	message(STATUS ">>---> Build : Release <---<<")
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/Release)
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/Release)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/Release)
endif()

##
## Add include and link paths.
##
include_directories(${CMAKE_SOURCE_DIR}/inc)
link_directories(${CMAKE_SOURCE_DIR}/lib)
add_definitions(-D_UNICODE -DUNICODE)

include_directories(${BOOST_INCLUDE_DIRS})
link_directories(${BOOST_LIBRARY_DIRS})
include_directories(${OPENCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#include_directories(/usr/local/include /usr/include)
#link_directories(/usr/local/lib /usr/lib)

##
## Set and add include and link paths for dev/sw_dev & dev/hw_dev.
##
if(WIN32 AND NOT CYGWIN)
#	set(MY_USR_LOCAL_BIN D:/usr/local/bin)
	set(MY_USR_LOCAL_INCLUDE D:/usr/local/include)
	set(MY_USR_LOCAL_LIB D:/usr/local/lib)
#	set(MY_USR_BIN D:/usr/bin)
#	set(MY_USR_INCLUDE D:/usr/include)
#	set(MY_USR_LIB D:/usr/lib)

	set(MY_SW_DEV_EXT_HOME D:/work/gdt_github/sw_dev/cpp/ext)
	set(MY_SW_DEV_RND_HOME D:/work/gdt_github/sw_dev/cpp/rnd)
	set(MY_HW_DEV_EXT_HOME D:/work/gdt_github/hw_dev/ext)

	set(MY_SW_DEV_EXT_INCLUDE ${MY_SW_DEV_EXT_HOME}/inc)
	set(MY_SW_DEV_EXT_LIB ${MY_SW_DEV_EXT_HOME}/lib)
	set(MY_SW_DEV_RND_INCLUDE ${MY_SW_DEV_RND_HOME}/inc)
	set(MY_SW_DEV_RND_LIB ${MY_SW_DEV_RND_HOME}/lib)
	set(MY_HW_DEV_EXT_INCLUDE ${MY_HW_DEV_EXT_HOME}/inc)
	set(MY_HW_DEV_EXT_LIB ${MY_HW_DEV_EXT_HOME}/lib)

#	include_directories(${MY_USR_LOCAL_INCLUDE} ${MY_USR_INCLUDE} ${MY_SW_DEV_EXT_INCLUDE} ${MY_SW_DEV_RND_INCLUDE})
	include_directories(${MY_USR_LOCAL_INCLUDE})
#	link_directories(${MY_USR_LOCAL_LIB} ${MY_USR_LIB} ${MY_SW_DEV_EXT_LIB} ${MY_SW_DEV_RND_LIB})
	link_directories(${MY_USR_LOCAL_LIB})
#elseif(APPLE)
#elseif(CYGWIN)
#elseif(MSYS)
#elseif(UNIX AND NOT APPLE AND NOT CYGWIN)
#elseif(MINGW)
#elseif(BORLAND)
#elseif(WATCOM)
#elseif(MSVC)
else()
#	set(MY_USR_LOCAL_BIN $ENV{HOME}/my_usr/local/bin)
	set(MY_USR_LOCAL_INCLUDE $ENV{HOME}/my_usr/local/include)
	set(MY_USR_LOCAL_LIB $ENV{HOME}/my_usr/local/lib)
#	set(MY_USR_BIN $ENV{HOME}/my_usr/bin)
#	set(MY_USR_INCLUDE $ENV{HOME}/my_usr/include)
#	set(MY_USR_LIB $ENV{HOME}/my_usr/lib)

	set(MY_SW_DEV_EXT_HOME $ENV{HOME}/work/gdt_github/sw_dev/cpp/ext)
	set(MY_SW_DEV_RND_HOME $ENV{HOME}/work/gdt_github/sw_dev/cpp/rnd)
	set(MY_HW_DEV_EXT_HOME $ENV{HOME}/work/gdt_github/hw_dev/ext)

	set(MY_SW_DEV_EXT_INCLUDE ${MY_SW_DEV_EXT_HOME}/inc)
	set(MY_SW_DEV_EXT_LIB ${MY_SW_DEV_EXT_HOME}/lib)
	set(MY_SW_DEV_RND_INCLUDE ${MY_SW_DEV_RND_HOME}/inc)
	set(MY_SW_DEV_RND_LIB ${MY_SW_DEV_RND_HOME}/lib)
	set(MY_HW_DEV_EXT_INCLUDE ${MY_HW_DEV_EXT_HOME}/inc)
	set(MY_HW_DEV_EXT_LIB ${MY_HW_DEV_EXT_HOME}/lib)

#	include_directories(${MY_USR_LOCAL_INCLUDE} ${MY_USR_INCLUDE} ${MY_SW_DEV_EXT_INCLUDE} ${MY_SW_DEV_RND_INCLUDE})
	include_directories(${MY_USR_LOCAL_INCLUDE})
#	link_directories(${MY_USR_LOCAL_LIB} ${MY_USR_LIB} ${MY_SW_DEV_EXT_LIB} ${MY_SW_DEV_RND_LIB})
	link_directories(${MY_USR_LOCAL_LIB})
endif()

##
## Install header files, data files, and binaries.
##
# FIXME [check & correct] >>
file(GLOB_RECURSE HEADER_FILES "${CMAKE_SOURCE_DIR}/inc/*.h")
install(FILES ${HEADER_FILES} DESTINATION include)
file(GLOB_RECURSE DATA_FILES "${CMAKE_SOURCE_DIR}/bin/data/*")
install(FILES ${DATA_FILES} DESTINATION bin/data)
#install(PROGRAMS test.sh DESTINATION bin)
install(TARGETS myExe myStaticLib mySharedLib
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
)

##
## Add sub directories to build.
##
#add_subdirectory(3rd)
add_subdirectory(src)
add_subdirectory(app)
if(BUILD_TESTS)
	add_subdirectory(test)
	add_subdirectory(unit)
endif(BUILD_TESTS)
if(BUILD_DOCS)
	add_subdirectory(doc)
endif(BUILD_DOCS)
