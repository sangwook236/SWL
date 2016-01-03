set(TARGET swl_winutil)

set(SRCS
#	MfcSockClient.cpp
	WinConsoleWindow.cpp
	WinFileCrypter.cpp
	WinFileLocker.cpp
	WinRegistrar.cpp
	WinSerialPort.cpp
	WinSockClient.cpp
)
set(LIBS
	swl_util
	swl_math
	swl_base
	Ws2_32
)

add_definitions(-DSWL_WIN_UTIL_EXPORT)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
