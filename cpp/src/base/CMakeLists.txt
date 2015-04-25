set(TARGET swl_base)
set(SRCS INotifier.cpp IObserver.cpp LogException.cpp ReturnException.cpp String.cpp)

add_definitions(-DSWL_BASE_EXPORT)

add_library(${TARGET} SHARED ${SRCS})
