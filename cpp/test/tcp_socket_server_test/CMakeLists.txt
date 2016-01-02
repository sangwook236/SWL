if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_tcp_socket_server_test_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_tcp_socket_server_test)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	EchoTcpSocketConnection.cpp
	EchoTcpSocketSession.cpp
	TcpSocketServerTest.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_util_d
		swl_base_d
#		${Boost_THREAD_LIBRARY}
#		${Boost_SYSTEM_LIBRARY}
		boost_thread
		boost_system
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_util
		swl_base
#		${Boost_THREAD_LIBRARY}
#		${Boost_SYSTEM_LIBRARY}
		boost_thread
		boost_system
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	add_compile_options(-fpermissive)
endif(CMAKE_CXX_COMPILER_ID MATCHES GNU)

add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})

add_test(test/${TARGET} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET})
