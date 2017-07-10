set(TARGET swl_util)

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
	WaveData.cpp
	xgetopt.cpp
#	xgetopt_example.cpp
)
set(LIBS
	swl_math
	swl_base
)

add_definitions(-DSWL_UTIL_EXPORT)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	add_compile_options(-fPIC)
endif(CMAKE_CXX_COMPILER_ID MATCHES GNU)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
