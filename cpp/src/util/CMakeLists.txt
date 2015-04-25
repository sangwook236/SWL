set(TARGET swl_util)
set(SRCS AsyncTcpSocketClient.cpp dictionary.cpp GuardedBuffer.cpp IniParser.cpp iniparser_impl.cpp PacketPacker.cpp RegionOfInterest.cpp RegionOfInterestMgr.cpp SerialPort.cpp strlib.cpp TcpSocketClient.cpp TcpSocketConnection.cpp TcpSocketSession.cpp xgetopt.cpp)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(LIBS swl_math swl_base log4cxx)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(LIBS swl_math swl_base log4cxx)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_definitions(-DSWL_UTIL_EXPORT)

add_library(${TARGET} SHARED ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
