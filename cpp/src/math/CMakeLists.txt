if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_math_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_math)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	ConvexHull.cpp
	Coordinates.cpp
	DataNormalization.cpp
	GeometryUtil.cpp
	MathConstant.cpp
	MathExt.cpp
	MathUtil.cpp
	Rational.cpp
	RootFinder.cpp
	Rotation.cpp
	Statistic.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_base_d
		${GSL_LIBRARIES}
		${LAPACK_LIBRARIES}
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_base
		${GSL_LIBRARIES}
		${LAPACK_LIBRARIES}
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_definitions(-DSWL_MATH_EXPORT)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
