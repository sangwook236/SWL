set(TARGET swl_data_analysis_tool)

set(SRCS
	main.cpp
)
set(LIBS
	swl_machine_vision
	swl_machine_learning
	swl_rnd_util
	${OpenCV_LIBRARIES}
)

add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
