set(TARGET swl_math)

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
set(LIBS
	swl_base
	${GSL_LIBRARIES}
	${LAPACK_LIBRARIES}
)

add_definitions(-DSWL_MATH_EXPORT)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	add_compile_options(-fPIC)
endif(CMAKE_CXX_COMPILER_ID MATCHES GNU)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
