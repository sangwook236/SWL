if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_winutil_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_winutil)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	MfcSockClient.cpp
	WinConsoleWindow.cpp
	WinFileCrypter.cpp
	WinFileLocker.cpp
	WinRegistrar.cpp
	WinSerialPort.cpp
	WinSockClient.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_util_d
		swl_math_d
		swl_base_d
		Ws2_32
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_util
		swl_math
		swl_base
		Ws2_32
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_definitions(-DSWL_WIN_UTIL_EXPORT)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
