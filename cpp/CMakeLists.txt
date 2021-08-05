cmake_minimum_required(VERSION 3.4)
project(SWL_CPP VERSION 1.0.0)

##
## Add path for custom modules.
##
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

##
## Set package information.
##
set(PACKAGE_NAME "SWL-C++: Sang-Wook's Library for C++")
set(PACKAGE_VERSION "1.0.0")
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(PACKAGE_WEBSITE "https://github.com/sangwook236/sangwook-library")
set(PACKAGE_BUGREPORT "sangwook236@gmail.com")

##
## Set platform specific options.
##	CMAKE_SYSTEM_NAME, CMAKE_SYSTEM
##
message(STATUS "[-- SWL-C++ --] Generator: ${CMAKE_GENERATOR}")
message(STATUS "[-- SWL-C++ --] Platform: ${CMAKE_SYSTEM_NAME} (${CMAKE_SYSTEM})")
if(APPLE)
	if(POLICY CMP0042)
		cmake_policy(SET CMP0042 NEW)
	endif()

	link_directories(/opt/local/lib)

	if("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")  # For Mac OS X.
	endif()
elseif(CYGWIN)
elseif(MSYS)
elseif(UNIX AND NOT APPLE AND NOT CYGWIN)
	if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")  # For Linux.
	endif()
elseif(WIN32 AND NOT CYGWIN)
elseif(MINGW)
elseif(BORLAND)
elseif(WATCOM)
elseif(MSVC)
#	add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
endif()

##
## Set compiler specific options.
##
message(STATUS "[-- SWL-C++ --] Compiler: ${CMAKE_CXX_COMPILER_ID}")
if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
	## Explicitly specify standard c++ library.
#	set(CLANG_LINK_FLAGS "-stdlib=libstdc++")

	## add_definitions() is needed to remove unknown warnings.
	add_definitions(${CLANG_LINK_FLAGS})
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CLANG_LINK_FLAGS}")
elseif(CMAKE_CXX_COMPILER_ID MATCHES GNU)
elseif(CMAKE_CXX_COMPILER_ID MATCHES Intel)
elseif(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
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
## Set build types: None, Debug, Release, MinSizeRel, and RelWithDebInfo.
##	CMAKE_BUILD_TYPE, CMAKE_CONFIGURATION_TYPES
##
##	if a command 'cmake -DCMAKE_BUILD_TYPE=Release ..' is executed
##		CMAKE_BUILD_TYPE=Release
##	else if a command 'cmake -DCMAKE_BUILD_TYPE=Debug ..' is executed
##		CMAKE_BUILD_TYPE=Debug
##	else
##		CMAKE_BUILD_TYPE=
##
message(STATUS "[-- SWL-C++ --] Build: ${CMAKE_BUILD_TYPE}")
message(STATUS "[-- SWL-C++ --] Build Configuration: ${CMAKE_CONFIGURATION_TYPES}")
set(CMAKE_DEBUG_POSTFIX _d)  # The CMAKE_DEBUG_POSTFIX variable only affects library targets.

##
## Set binary directories.
##
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

if("${CMAKE_GENERATOR}" STRGREATER "Visual Studio")
	foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
		string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
		set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
		set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
		set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
	endforeach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
endif()

##
## Set options.
##
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_DOCS "Build documents" OFF)

if(BUILD_SHARED_LIBS)
	set(LIB_TYPE SHARED)
else(BUILD_SHARED_LIBS)
	set(LIB_TYPE STATIC)
endif(BUILD_SHARED_LIBS)

if(BUILD_TESTS)
	enable_testing()
	set(USE_UNIT_TESTCASES_ONLY_AS_TESTS ON)
endif(BUILD_TESTS)

#option(USE_THREAD "Thread-safe build (using pthreads)" ON)
set(USE_THREAD ON)
#option(USE_OPENGL "OpenGL" ON)
set(USE_OPENGL ON)
#option(USE_GLUT "OpenGL Utility Toolkit (GLUT)" ON)
set(USE_GLUT ON)
#option(USE_BOOST "Boost C++ Libraries" ON)
set(USE_BOOST ON)

#option(USE_LOG4CXX "Apache log4cxx - a logging framework for C++" ON)
set(USE_LOG4CXX ON)
#option(USE_GLOG "glog - C++ implementation of the Google logging module" OFF)
set(USE_GLOG OFF)
#option(USE_CPPUNIT "CppUnit - C++ port of JUnit" OFF)  # deprecated.
set(USE_CPPUNIT OFF)
option(USE_GOOGLE_TEST "Google Test - Google's C++ test framework" OFF)
#set(USE_GOOGLE_TEST OFF)
#option(USE_GOOGLE_MOCK "Google Mock - Google C++ Mocking Framework" OFF)
set(USE_GOOGLE_MOCK OFF)

#option(USE_LAPACK "Linear Algebra PACKage (LAPACK)" ON)
set(USE_LAPACK ON)
#option(USE_EIGEN "Eigen - C++ template library for linear algebra" ON)
set(USE_EIGEN ON)

#option(USE_GSL "GNU Scientific Library (GSL)" ON)
set(USE_GSL ON)
#option(USE_OPENCV "Open Source Computer Vision (OpenCV)" OFF)
#set(USE_OPENCV OFF)
#option(USE_OPENCV2 "Open Source Computer Vision (OpenCV) ver 2" ON)
set(USE_OPENCV2 ON)

#option(USE_FFTW "Fastest Fourier Transform in the West (FFTW)" OFF)
set(USE_FFTW OFF)
#option(USE_PCL "Point Cloud Library (PCL)" OFF)
set(USE_PCL OFF)

option(USE_CUDA "Compute Unified Device Architecture (CUDA) - a parallel computing platform and programming model invented by NVIDIA" OFF)
#set(USE_CUDA OFF)

#option(USE_QT4 "Qt4 - Coss-platform application framework" OFF)
set(USE_QT4 OFF)
#option(USE_QT5 "Qt5 - Coss-platform application framework" OFF)
set(USE_QT5 OFF)

option(USE_DOXYGEN "Doxygen - De facto standard tool for generating documentation from annotated C++ sources" ON)
#set(USE_DOXYGEN ON)
#option(USE_LYX "LyX - Document Processor" OFF)
set(USE_LYX OFF)

##
## Find packages.
##
if(USE_THREAD)
	# Variables: CMAKE_THREAD_LIBS_INIT, CMAKE_USE_SPROC_INIT, CMAKE_USE_WIN32_THREADS_INIT, CMAKE_USE_PTHREADS_INIT, CMAKE_HP_PTHREADS_INIT
	set(CMAKE_THREAD_PREFER_PTHREAD ON)  # For pthreads.
	find_package(Threads REQUIRED)  # target_link_libraries(myexe ${CMAKE_THREAD_LIBS_INIT})

	if(CMAKE_USE_SPROC_INIT)
		message(STATUS "[-- SWL-C++ --] Thread: sproc")
	elseif(CMAKE_USE_WIN32_THREADS_INIT)
		message(STATUS "[-- SWL-C++ --] Thread: Win32 Threads")
	elseif(CMAKE_USE_PTHREADS_INIT)
		message(STATUS "[-- SWL-C++ --] Thread: pthreads")
		add_definitions(-D_REENTRANT)
	elseif(CMAKE_HP_PTHREADS_INIT)
		message(STATUS "[-- SWL-C++ --] Thread: HP pthreads")
	endif()
endif(USE_THREAD)

if(USE_OPENGL)
	# Variables: OPENGL_FOUND, OPENGL_XMESA_FOUND, OPENGL_GLU_FOUND, OPENGL_INCLUDE_DIR, OPENGL_LIBRARIES
	find_package(OpenGL REQUIRED)
	include_directories(${OPENGL_INCLUDE_DIR})
endif(USE_OPENGL)
if(USE_GLUT)
	# Variables: GLUT_FOUND, GLUT_INCLUDE_DIR, GLUT_LIBRARIES
	find_package(GLUT REQUIRED)
	include_directories(${GLUT_INCLUDE_DIR})
endif(USE_GLUT)

if(USE_BOOST)
	# Hints: BOOST_ROOT, BOOST_INCLUDEDIR, BOOST_LIBRARYDIR
	# Variables: Boost_FOUND, Boost_INCLUDE_DIRS, Boost_LIBRARY_DIRS, Boost_LIBRARIES, Boost_<C>_FOUND, Boost_<C>_LIBRARY, Boost_VERSION, Boost_LIB_VERSION, Boost_MAJOR_VERSION, Boost_MINOR_VERSION, Boost_SUBMINOR_VERSION
	#find_package(Boost 1.62 REQUIRED)
	find_package(Boost 1.62 REQUIRED thread unit_test_framework timer)
	include_directories(${Boost_INCLUDE_DIRS})
	link_directories(${Boost_LIBRARY_DIRS})

	# FIXME [check] >>
	if(Boost_FOUND)
		if(NOT Boost_THREAD_FOUND OR "${Boost_THREAD_LIBRARY}" STREQUAL "")
			find_package(Boost COMPONENTS thread)
		endif()
		if(NOT Boost_UNIT_TEST_FRAMEWORK_FOUND OR "${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}" STREQUAL "")
			find_package(Boost COMPONENTS unit_test_framework)
		endif()
		if(NOT Boost_TIMER_FOUND OR "${Boost_TIMER_LIBRARY}" STREQUAL "")
			find_package(Boost COMPONENTS timer)
		endif()
	endif(Boost_FOUND)
endif(USE_BOOST)

if(USE_LOG4CXX)
	# ${SWL_CPP_HOME}/cmake/FindLog4cxx.cmake
	# Variables: LOG4CXX_FOUND, LOG4CXX_INCLUDE_DIR, LOG4CXX_LIBRARIES
	find_package(Log4cxx REQUIRED)
	include_directories(${LOG4CXX_INCLUDE_DIR})
endif(USE_LOG4CXX)
if(USE_GLOG)
	# Variables: 
endif(USE_GLOG)
if(USE_CPPUNIT)
	# Variables: CPPUNIT_FOUND, CPPUNIT_INCLUDE_DIR, CPPUNIT_LIBRARIES
	find_package(CPPUNIT)
	include_directories(${CPPUNIT_INCLUDE_DIR})
endif(USE_CPPUNIT)
if(USE_GOOGLE_TEST)
	# Variables: GTEST_FOUND, GTEST_INCLUDE_DIRS, GTEST_BOTH_LIBRARIES, GTEST_LIBRARIES, GTEST_MAIN_LIBRARIES
	find_package(GTest)
	include_directories(${GTEST_INCLUDE_DIRS})
endif(USE_GOOGLE_TEST)
if(USE_GOOGLE_MOCK)
	# Variables: 
	find_package(GMock)
endif(USE_GOOGLE_MOCK)

if(USE_LAPACK)
	if(APPLE)
		# Variables: LAPACKLIBS_VECLIB_FRAMEWORK, LAPACKLIBS_ACCELERATE_FRAMEWORK, LAPACK_LIBRARIES, BLAS_LIBRARIES
		find_package(vecLib REQUIRED)
#		find_package(LAPACK REQUIRED)

		if(NOT LAPACK_LIBRARIES_DIR)
			set(ATLAS_LIBRARIES_DIR /opt/local/lib)
		else()
			set(ATLAS_LIBRARIES_DIR ${LAPACK_LIBRARIES_DIR})
		endif()
		if(NOT LAPACK_INCLUDE_DIR)
			set(ATLAS_INCLUDE_DIR /opt/local/include)
		else()
			set(ATLAS_INCLUDE_DIR ${LAPACK_LIBRARIES_DIR})
		endif()
	else(APPLE)
		# Variables: LAPACK_FOUND, LAPACK_LINKER_FLAGS, LAPACK_LIBRARIES, LAPACK95_LIBRARIES, LAPACK95_FOUND, BLA_STATIC, BLA_VENDOR, BLA_F95
		find_package(LAPACK REQUIRED)

		if(NOT LAPACK_LIBRARIES_DIR)
			set(ATLAS_LIBRARIES_DIR /usr/lib/atlas-base)  # for Ubuntu.
#			set(ATLAS_LIBRARIES_DIR /usr/lib64/atlas)  # for RHEL.
		else()
			set(ATLAS_LIBRARIES_DIR ${LAPACK_LIBRARIES_DIR})
		endif()
		if(NOT LAPACK_INCLUDE_DIR)
			set(ATLAS_INCLUDE_DIR /usr/include/atlas)  # for Ubuntu.
#			set(ATLAS_INCLUDE_DIR /usr/include)  # for RHEL.
		else()
			set(ATLAS_INCLUDE_DIR ${LAPACK_LIBRARIES_DIR})
		endif()
	endif(APPLE)
endif(USE_LAPACK)
if(USE_EIGEN)
	# ${SWL_CPP_HOME}/cmake/FindEigen.cmake
	# Variables: EIGEN_FOUND, EIGEN_VERSION, EIGEN_INCLUDE_DIRS
	find_package(Eigen 3.2 REQUIRED)
	include_directories(${EIGEN_INCLUDE_DIRS})
endif(USE_EIGEN)

if(USE_GSL)
	# Variables: GSL_FOUND, GSL_LIBRARIES, GSL_VERSION, GSL_ROOT_DIR
	find_package(GSL REQUIRED)
	include_directories(${GSL_INCLUDE_DIRS})
endif(USE_GSL)
if(USE_OPENCV)
	# ${SWL_CPP_HOME}/cmake/FindOpenCVcmake
	# Variables: OpenCV_FOUND, OpenCV_VERSION_MAJOR, OpenCV_VERSION_MINOR, OpenCV_VERSION_PATCH, OpenCV_INCLUDE_DIRS, OpenCV_LIBS, OpenCV_LIB_DIR
	find_package(OpenCV REQUIRED)
	include_directories(${OpenCV_INCLUDE_DIRS})
endif(USE_OPENCV)
if(USE_OPENCV2)
	# ${SWL_CPP_HOME}/cmake/FindOpenCV2cmake
	# Hints: OpenCV2_ROOT_DIR
	# Variables: OpenCV2_FOUND, OpenCV2_INCLUDE_DIRS, OpenCV2_LIBRARIES
	find_package(OpenCV2 3.2.0 REQUIRED)
	include_directories(${OpenCV2_INCLUDE_DIRS})

#	message(STATUS "[-- SWL-C++ --] OpenCV: ${OpenCV2_LIBRARIES}")
endif(USE_OPENCV2)

if(USE_FFTW)
	# ${SWL_CPP_HOME}/cmake/FindFFTW.cmake
	# Variables: FFTW_FOUND, FFTW_INCLUDES, FFTW_LIBRARIES
	find_package(FFTW)
	include_directories(${FFTW_INCLUDES})
endif(USE_FFTW)
if(USE_PCL)
	find_package(PCL 1.5)
	include_directories(${PCL_INCLUDE_DIRS})
	link_directories(${PCL_LIBRARY_DIRS})
	add_definitions(${PCL_DEFINITIONS})
endif(USE_PCL)

if(USE_CUDA)
	# Variables: CUDA_FOUND, CUDA_VERSION, CUDA_VERSION_STRING, CUDA_TOOLKIT_ROOT_DIR, CUDA_SDK_ROOT_DIR, CUDA_INCLUDE_DIRS, CUDA_LIBRARIES, CUDA_CUFFT_LIBRARIES, CUDA_CUBLAS_LIBRARIES, CUDA_cupti_LIBRARY, CUDA_curand_LIBRARY, CUDA_cusparse_LIBRARY, CUDA_npp_LIBRARY, CUDA_nppc_LIBRARY, CUDA_nppi_LIBRARY, CUDA_npps_LIBRARY, CUDA_nvcuvenc_LIBRARY, CUDA_nvcuvid_LIBRARY
	find_package(CUDA)
	include_directories(${CUDA_INCLUDE_DIRS})

#	message(STATUS "[-- SWL-C++ --] CUDA: ${CUDA_LIBRARIES}")
endif(USE_CUDA)

#if(USE_QT4)
#	# Variables: QT_FOUND, Qt4_FOUND, QT_VERSION_MAJOR, QT_VERSION_MINOR, QT_VERSION_PATCH, QT_INCLUDE_DIR, QT_LIBRARIES, QT_DEFINITIONS
#	find_package(Qt4 4.7.4 QUIETLY QtGui QtXml)  # target_link_libraries(myexe Qt4::QtGui Qt4::QtXml)
#	include_directories(${QT_INCLUDE_DIR})
#	add_definitions(${QT_DEFINITIONS})
#endif(USE_QT4)
if(USE_QT5)
	# Variables: QT_FOUND, Qt5_FOUND, QT_VERSION_MAJOR, QT_VERSION_MINOR, QT_VERSION_PATCH, QT_INCLUDE_DIR, QT_LIBRARIES, QT_DEFINITIONS
	# Variables: Qt5Core_INCLUDE_DIRS, Qt5Core_LIBRARIES
	find_package(Qt5 COMPONENTS Core Gui Widgets Quick Qml Concurrent REQUIRED)  # target_link_libraries(myexe Qt5::Qt5Core Qt5::Qt5Widgets)
	#find_package(Qt5Core REQUIRED)
	#find_package(Qt5Gui REQUIRED)
	#find_package(Qt5Widgets REQUIRED)
	#find_package(Qt5Quick REQUIRED)
	#find_package(Qt5Qml REQUIRED)
	#find_package(Qt5Concurrent REQUIRED)
	include_directories(${QT_INCLUDE_DIR})
	add_definitions(${QT_DEFINITIONS})
endif(USE_QT5)

if(USE_DOXYGEN)
	# Variables: DOXYGEN_FOUND, DOXYGEN_VERSION, DOXYGEN_EXECUTABLE
	# Variables: DOXYGEN_DOT_FOUND, DOXYGEN_DOT_PATH, DOXYGEN_DOT_EXECUTABLE
	find_package(Doxygen)
endif(USE_DOXYGEN)
if(USE_LYX)
	# ${SWL_CPP_HOME}/cmake/FindLyX.cmake
	# Variables: LYX_FOUND, LYX_COMMAND
	find_package(LyX)
	set(LYX_EXECUTABLE lyx)
endif(USE_LYX)

##
## Add include and link paths.
##
include_directories(${CMAKE_SOURCE_DIR}/inc)  # ../../inc
link_directories(${CMAKE_SOURCE_DIR}/lib)  # ../../lib
add_definitions(-D_UNICODE -DUNICODE)

#include_directories(/usr/local/include /usr/include)
#link_directories(/usr/local/lib /usr/lib)

##
## Set and add include and link paths for swdt/sw_dev & swdt/hw_dev.
##
if(WIN32 AND NOT CYGWIN)
#	set(MY_USR_LOCAL_BIN D:/usr/local/bin)
	set(MY_USR_LOCAL_INCLUDE D:/usr/local/include)
	set(MY_USR_LOCAL_LIB D:/usr/local/lib)
#	set(MY_USR_BIN D:/usr/bin)
#	set(MY_USR_INCLUDE D:/usr/include)
#	set(MY_USR_LIB D:/usr/lib)

	set(MY_SW_DEV_EXT_HOME D:/work/swdt_github/sw_dev/cpp/ext)
	set(MY_SW_DEV_RND_HOME D:/work/swdt_github/sw_dev/cpp/rnd)
	set(MY_HW_DEV_EXT_HOME D:/work/swdt_github/hw_dev/ext)

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

	set(MY_SW_DEV_EXT_HOME $ENV{HOME}/work/swdt_github/sw_dev/cpp/ext)
	set(MY_SW_DEV_RND_HOME $ENV{HOME}/work/swdt_github/sw_dev/cpp/rnd)
	set(MY_HW_DEV_EXT_HOME $ENV{HOME}/work/swdt_github/hw_dev/ext)

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
file(GLOB EXECUTABLE_FILES "${CMAKE_SOURCE_DIR}/bin/*")
#install(PROGRAMS ${EXECUTABLE_FILES} DESTINATION bin)
#install(TARGETS myExe myStaticLib mySharedLib
#	RUNTIME DESTINATION bin
#	LIBRARY DESTINATION lib
#	ARCHIVE DESTINATION lib
#)

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
