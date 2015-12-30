include_directories(./efficient_graph_based_image_segmentation_lib ${MY_USR_LOCAL_INCLUDE}/opengm_external/maxflow)

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_kinect_segmentation_app_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_kinect_segmentation_app)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(EFFICIENT_GRAPH_BASED_IMAGE_SEGMENTATION_LIB_SRCS
#	./efficient_graph_based_image_segmentation_lib/segment.cpp
)
set(GSLIC_LIB_SRCS
	./gslic_lib/cudaImgTrans.cu
	./gslic_lib/cudaSegEngine.cu
	./gslic_lib/cudaSegSLIC.cu
	./gslic_lib/cudaUtil.cu
	./gslic_lib/FastImgSeg.cpp
)

set(SRCS
	${EFFICIENT_GRAPH_BASED_IMAGE_SEGMENTATION_LIB_SRCS}
	${GSLIC_LIB_SRCS}
	main.cpp
	DepthGuidedMap.cpp
	EfficientGraphBasedImageSegmentation.cpp
	gSLIC.cpp
	NonMaximaSuppression.cpp
	SegmentationUsingGrabCut.cpp
	SegmentationUsingGraphCut.cpp
	Util.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_machine_vision_d
#		opengm_min_sum
#		opengm_min_sum_small
		external-library-mrf_d
		external-library-maxflow_d
		external-library-maxflow-ibfs_d
		external-library-qpbo_d
		external-library-trws_d
		${OpenCV_LIBS}
		${CUDA_LIBRARIES}
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_machine_vision
#		opengm_min_sum
#		opengm_min_sum_small
		external-library-mrf
		external-library-maxflow
		external-library-maxflow-ibfs
		external-library-qpbo
		external-library-trws
		${OpenCV_LIBS}
		${CUDA_LIBRARIES}
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
