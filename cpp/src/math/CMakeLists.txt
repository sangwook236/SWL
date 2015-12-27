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
	set(LIBS
		opencv_calib3d243d
		opencv_contrib243d
		opencv_core243d
		opencv_features2d243d
		opencv_flann243d
		opencv_gpu243d
		opencv_highgui243d
		opencv_imgproc243d
		opencv_legacy243d
		opencv_ml243d
		opencv_nonfree243d
		opencv_objdetect243d
		opencv_photo243d
		opencv_stitching243d
		opencv_ts243d
		opencv_video243d
		opencv_videostab243d
		swl_base
		swl_math
		cblas
		gsl
		log4cxx
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		opencv_calib3d243
		opencv_contrib243
		opencv_core243
		opencv_features2d243
		opencv_flann243
		opencv_gpu243
		opencv_highgui243
		opencv_imgproc243
		opencv_legacy243
		opencv_ml243
		opencv_nonfree243
		opencv_objdetect243
		opencv_photo243
		opencv_stitching243
		opencv_ts243
		opencv_video243
		opencv_videostab243
		swl_base
		swl_math
		cblas
		gsl
		log4cxx
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_definitions(-DSWL_MATH_EXPORT)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
