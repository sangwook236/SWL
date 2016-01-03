set(TARGET swl_base)

set(SRCS
	INotifier.cpp
	IObserver.cpp
	LogException.cpp
	ReturnException.cpp
	String.cpp
)
set(LIBS
	${LOG4CXX_LIBRARIES}
)

add_definitions(-DSWL_BASE_EXPORT)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	add_compile_options(-fPIC)
endif(CMAKE_CXX_COMPILER_ID MATCHES GNU)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
