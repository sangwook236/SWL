set(TARGET swl_serial_port_test)

set(SRCS
	main.cpp
	serial_port_test.cpp
)
set(LIBS
	swl_winutil
	swl_util
	swl_base
)

add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})

add_test(
	NAME unit_test/${TARGET}
	COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}
	WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
