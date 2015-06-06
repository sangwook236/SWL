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
set(PACKAGE_VERSION "1.0")
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(PACKAGE_WEBSITE "http://www.sangwook.com/")
set(PACKAGE_BUGREPORT "sangwook236@gmail.com")

##
## Find packages.
##
find_package(Threads REQUIRED)
find_package(Boost 1.52 REQUIRED)
find_package(GSL REQUIRED)
find_package(OpenCV REQUIRED)
#find_package(PCL 1.5 REQUIRED)

find_package(Doxygen)
#find_package(LyX)  # error : By not providing "FindLyX.cmake" in CMAKE_MODULE_PATH ...
#set(LYX_FOUND NO)  # if LyX is not found or does not try being found, LYX_FOUND is set to NO.
#set(LYX_FOUND YES)
#set(LYX_EXECUTABLE lyx)

##
## Set platform specific options.
##
if(APPLE)
	message(STATUS ">>-—-> Platform : Apple <---<<")

	if(POLICY CMP0042)
		cmake_policy(SET CMP0042 NEW)
	endif()

	link_directories(/opt/local/lib)

	if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")  # for Mac OS X.
	endif()
elseif(CYGWIN)
	message(STATUS ">>-—-> Platform : Cygwin <---<<")
elseif(MSYS)
	message(STATUS ">>-—-> Platform : MSYS <---<<")
elseif(UNIX AND NOT APPLE AND NOT CYGWIN)
	message(STATUS ">>-—-> Platform : Unix <---<<")
elseif(WIN32 AND NOT CYGWIN)
	message(STATUS ">>-—-> Platform : WIN32 <---<<")
elseif(MINGW)
	message(STATUS ">>-—-> Platform : MinGW <---<<")
elseif(BORLAND)
	message(STATUS ">>-—-> Platform : Borland <---<<")
elseif(WATCOM)
	message(STATUS ">>-—-> Platform : Watcom <---<<")
elseif(MSVC)
	message(STATUS ">>-—-> Platform : MSVC <---<<")
endif()

##
## Set compiler specific options.
##
if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
	message(STATUS ">>-—-> Compiler : Clang <---<<")

	## Explicitly specify standard c++ library.
	#set(CLANG_LINK_FLAGS "-stdlib=libstdc++")

	## add_definitions() is needed to remove unknown warnings.
	add_definitions(${CLANG_LINK_FLAGS})
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CLANG_LINK_FLAGS}")
elseif(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	message(STATUS ">>-—-> Compiler : GNU <---<<")
elseif(CMAKE_CXX_COMPILER_ID MATCHES Intel)
	message(STATUS ">>-—-> Compiler : Intel <---<<")
elseif(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
	message(STATUS ">>-—-> Compiler : MSVC <---<<")

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
	message(STATUS ">>-—-> Build : Debug <---<<")
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/Debug)
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/Debug)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/Debug)
#elseif(CMAKE_BUILD_TYPE MATCHES RELEASE)
else()
	message(STATUS ">>-—-> Build : Release <---<<")
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
	set(MY_DEV_SW_EXT_HOME /usr/local)
	set(MY_DEV_SW_RND_HOME /usr/local)
	set(MY_DEV_HW_EXT_HOME /usr/local)
#elseif(APPLE)
#elseif(CYGWIN)
#elseif(MSYS)
#elseif(UNIX AND NOT APPLE AND NOT CYGWIN)
#elseif(MINGW)
#elseif(BORLAND)
#elseif(WATCOM)
#elseif(MSVC)
else()
	set(MY_DEV_SW_EXT_HOME $ENV{HOME}/my_usr/local)
	set(MY_DEV_SW_RND_HOME $ENV{HOME}/my_usr/local)
	set(MY_DEV_HW_EXT_HOME $ENV{HOME}/my_usr/local)
endif()

set(MY_DEV_SW_EXT_INCLUDE ${MY_DEV_SW_EXT_HOME}/include)
set(MY_DEV_SW_EXT_LIB ${MY_DEV_SW_EXT_HOME}/lib)
set(MY_DEV_SW_RND_INCLUDE ${MY_DEV_SW_RND_HOME}/include)
set(MY_DEV_SW_RND_LIB ${MY_DEV_SW_RND_HOME}/lib)
set(MY_DEV_HW_EXT_INCLUDE ${MY_DEV_HW_EXT_HOME}/include)
set(MY_DEV_HW_EXT_LIB ${MY_DEV_HW_EXT_HOME}/lib)

#include_directories(${MY_DEV_SW_EXT_INCLUDE} ${MY_DEV_SW_RND_INCLUDE} ${MY_DEV_HW_EXT_INCLUDE})
include_directories(${MY_DEV_SW_EXT_INCLUDE} ${MY_DEV_SW_RND_INCLUDE})
#link_directories(${MY_DEV_SW_EXT_LIB} ${MY_DEV_SW_RND_LIB} ${MY_DEV_HW_EXT_LIB})
link_directories(${MY_DEV_SW_EXT_LIB} ${MY_DEV_SW_RND_LIB})

##
## Set options.
##
option(BUILD_TESTS "Build tests" ON)
option(BUILD_DOCS "Build documents" ON)

if(BUILD_TESTS)
	enable_testing()
endif(BUILD_TESTS)

##
## Adding sub directories to build.
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
