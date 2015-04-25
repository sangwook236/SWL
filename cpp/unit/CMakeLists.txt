#add_subdirectory(testcase)

set(BASE_SRCS testcase/base/LoggerTest.cpp)

set(TARGET swl_unit_test)
set(SRCS text_ui/TextUiUnitTestMain.cpp ${BASE_SRCS})
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(
        LIBS
#       swl_winview swl_winutil swl_view.lib swl_util.lib swl_glutil.lib swl_math.lib swl_base.lib log4cxx.lib
        swl_base log4cxx.lib
    )
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(
        LIBS
#       swl_winview swl_winutil swl_view.lib swl_util.lib swl_glutil.lib swl_math.lib swl_base.lib log4cxx.lib
        swl_base log4cxx.lib
    )
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_definitions(-D__SWL_UNIT_TEST__USE_BOOST_UNIT)

add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})

add_test(swl_unit_test ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET})
