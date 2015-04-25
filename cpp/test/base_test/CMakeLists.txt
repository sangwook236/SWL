set(TARGET swl_base_test)
set(SRCS main.cpp)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(LIBS swl_base)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(LIBS swl_base)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})

add_test(test/${TARGET} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET})
