if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_util_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_util)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	AsyncTcpSocketClient.cpp
	dictionary.cpp
	GuardedBuffer.cpp
#	iniexample.cpp
	IniParser.cpp
	iniparser_impl.cpp
	PacketPacker.cpp
	RegionOfInterest.cpp
	RegionOfInterestMgr.cpp
	SerialPort.cpp
	strlib.cpp
	TcpSocketClient.cpp
	TcpSocketConnection.cpp
	TcpSocketSession.cpp
	xgetopt.cpp
#	xgetopt_example.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_math_d
		swl_base_d
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_math
		swl_base
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_definitions(-DSWL_UTIL_EXPORT)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
