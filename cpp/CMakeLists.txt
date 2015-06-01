project(SWL_CPP)
cmake_minimum_required(VERSION 2.8)

##
## Add path for custom modules.
##
set(
    CMAKE_MODULE_PATH
	${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake"
)

##
## Set package information.
##
set(PACKAGE_NAME "Sang-Wook's Library for C++")
set(PACKAGE_VERSION "1.0")
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(PACKAGE_WEBSITE "http://www.sangwook.com/")
set(PACKAGE_BUGREPORT "sangwook236@gmail.com")

##
## Set compiler specific options.
##
if(${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
	##
	## Explicitly specify standard c++ library.
	##
	SET(CLANG_LINK_FLAGS "-stdlib=libstdc++")
	add_definitions(${CLANG_LINK_FLAGS})
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CLANG_LINK_FLAGS}")
	## add_definitions() is needed to remove unknown warnings.
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
	SET(CMAKE_MFC_FLAG 2)  # Use MFC in a Shared DLL.
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
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

##
## Find packages.
##
find_package(Threads REQUIRED)
find_package(Boost 1.52 REQUIRED)
find_package(OpenCV REQUIRED)
#find_package(PCL 1.5 REQUIRED)

find_package(Doxygen)
#find_package(LyX)  # error : By not providing "FindLyX.cmake" in CMAKE_MODULE_PATH ...
#set(LYX_FOUND NO)  # if LyX is not found or does not try being found, LYX_FOUND is set to NO.
#set(LYX_FOUND YES)
#set(LYX_EXECUTABLE lyx)

##
## Set options.
##
option(BUILD_TESTS "Build tests" ON)
option(BUILD_DOCS "Build documents" ON)

if (BUILD_TESTS)
	enable_testing()
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

##
## Set and add include and link paths for dev/sw_dev & dev/hw_dev.
##
set(SW_DEV_SW_EXT_INCLUDE /usr/local/include)
set(SW_DEV_SW_EXT_LIB /usr/local/lib)
set(SW_DEV_SW_RND_INCLUDE /usr/local/include)
set(SW_DEV_SW_RND_LIB /usr/local/lib)
set(SW_DEV_HW_EXT_INCLUDE /usr/local/include)
set(SW_DEV_HW_EXT_LIB /usr/local/lib)
set(SW_DEV_HW_RND_INCLUDE /usr/local/include)
set(SW_DEV_HW_RND_LIB /usr/local/lib)

include_directories(${SW_DEV_SW_EXT_INCLUDE} ${SW_DEV_SW_RND_INCLUDE})
link_directories(${SW_DEV_SW_EXT_LIB} ${SW_DEV_SW_RND_LIB})

##
## Adding sub directories to build.
##
#add_subdirectory(3rd)
add_subdirectory(src)
add_subdirectory(app)
if (BUILD_TESTS)
	add_subdirectory(test)
	add_subdirectory(unit)
endif(BUILD_TESTS)
if (BUILD_DOCS)
	add_subdirectory(doc)
endif(BUILD_DOCS)
