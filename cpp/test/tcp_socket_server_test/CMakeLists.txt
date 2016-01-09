set(TARGET swl_tcp_socket_server_test)

set(SRCS
	EchoTcpSocketConnection.cpp
	EchoTcpSocketSession.cpp
	TcpSocketServerTest.cpp
)
set(LIBS
	swl_util
	swl_base
	${Boost_THREAD_LIBRARY}
	${Boost_SYSTEM_LIBRARY}
)

add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})

if(NOT USE_UNIT_TESTCASES_ONLY_AS_TESTS)
	add_test(
		NAME test/${TARGET}
		COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}
		WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
	)
endif(NOT USE_UNIT_TESTCASES_ONLY_AS_TESTS)
