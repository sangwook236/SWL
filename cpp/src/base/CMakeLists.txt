if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_base_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_base)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	INotifier.cpp
	IObserver.cpp
	LogException.cpp
	ReturnException.cpp
	String.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		log4cxx_d
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		log4cxx
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_definitions(-DSWL_BASE_EXPORT)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
