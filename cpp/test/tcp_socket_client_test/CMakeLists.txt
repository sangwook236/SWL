if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_tcp_socket_client_test_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_tcp_socket_client_test)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	TcpSocketClientTest.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_util_d
		swl_base_d
		${Boost_THREAD_LIBRARY}
		${Boost_SYSTEM_LIBRARY}
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_util
		swl_base
		${Boost_THREAD_LIBRARY}
		${Boost_SYSTEM_LIBRARY}
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})

add_test(
	NAME unit_test/${TARGET}
	COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}
	WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
