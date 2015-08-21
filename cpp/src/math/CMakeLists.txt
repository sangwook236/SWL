set(TARGET swl_math)
set(SRCS
	ConvexHull.cpp
	Coordinates.cpp
	GeometryUtil.cpp
	MathConstant.cpp
	MathExt.cpp
	MathUtil.cpp
	Rational.cpp
	RootFinder.cpp
	Rotation.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS swl_base log4cxx)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS swl_base log4cxx)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_definitions(-DSWL_MATH_EXPORT)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
