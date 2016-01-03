set(TARGET swl_tcp_socket_client_test)

set(SRCS
	TcpSocketClientTest.cpp
)
set(LIBS
	swl_util
	swl_base
	${Boost_THREAD_LIBRARY}
	${Boost_SYSTEM_LIBRARY}
)

add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})

add_test(
	NAME unit_test/${TARGET}
	COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}
	WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
