include_directories(
	./efficient_graph_based_image_segmentation_lib
	${MY_USR_LOCAL_INCLUDE}/opengm_external/maxflow
)

set(TARGET swl_kinect_segmentation_app)

set(SRCS
	# efficient graph based image segmentation.
#	./efficient_graph_based_image_segmentation_lib/segment.cpp

	# gSLIC.
#	./gslic_lib/FastImgSeg.cpp
	./gslic_lib/cudaImgTrans.cu
	./gslic_lib/cudaSegEngine.cu
	./gslic_lib/cudaSegSLIC.cu
	./gslic_lib/cudaUtil.cu

	main.cpp
	DepthGuidedMap.cpp
	EfficientGraphBasedImageSegmentation.cpp
	gSLIC.cpp
	SegmentationUsingGrabCut.cpp
	SegmentationUsingGraphCut.cpp
	Util.cpp
)
set(LIBS
	swl_machine_vision
#	${OpenGM_LIBRARIES}
#	opengm_min_sum
#	opengm_min_sum_small
	external-library-mrf
	external-library-maxflow
	external-library-maxflow-ibfs
	external-library-qpbo
	external-library-trws
	${OpenCV2_LIBRARIES}
	${CUDA_LIBRARIES}
)

list(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)

#add_executable(${TARGET} ${SRCS})
cuda_add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
