# To find OpenCV 2 library visit http://opencv.willowgarage.com/wiki/
#
# The follwoing variables are optionally searched for defaults
#  OpenCV2_ROOT_DIR:                   Base directory of OpenCV 2 tree to use.
#
# The following are set after configuration is done:
#  OpenCV2_FOUND
#  OpenCV2_INCLUDE_DIRS
#  OpenCV2_LIBRARIES
#
# $Id: $
#  
# Balazs [2011-01-18]:
# - Created from scratch for the reorganized OpenCV 2 structure introduced at version 2.2
# Jbohren [2011-06-10]:
# - Added OpenCV_ROOT_DIR for UNIX platforms & additional opencv include dir
# jmorrison [2013-11-14]:
# - Added flag to disable GPU requirement (NO_OPENCV_GPU)
# 
# This file should be removed when CMake will provide an equivalent

#--- Select exactly ONE OpenCV 2 base directory to avoid mixing different version headers and libs
find_path(OpenCV2_ROOT_INC_DIR NAMES opencv2/opencv.hpp
    PATHS
        #--- WINDOWS
        C:/Developer/include                # Windows
        "$ENV{OpenCV_ROOT_DIR}/include"     # *NIX: custom install
        /usr/local/include                  # Linux: default dir by CMake
        /usr/include                        # Linux
        /opt/local/include                  # OS X: default MacPorts location
        NO_DEFAULT_PATH)

#--- DEBUG
#message(STATUS "OpenCV2_ROOT_INC_DIR: ${OpenCV2_ROOT_INC_DIR}")

#--- OBSOLETE
# Get parent of OpenCV2_ROOT_INC_DIR. We do this as it is more
# reliable than finding include/opencv2/opencv.hpp directly.
#GET_FILENAME_COMPONENT(OpenCV2_ROOT_DIR ${OpenCV2_ROOT_INC_DIR} PATH)
#message(STATUS "OpenCV2_ROOT_DIR: ${OpenCV2_ROOT_DIR}")

find_path(OpenCV2_CORE_INCLUDE_DIR       	NAMES core.hpp			PATHS "${OpenCV2_ROOT_INC_DIR}/opencv2/core")
find_path(OpenCV2_IMGPROC_INCLUDE_DIR    	NAMES imgproc.hpp		PATHS "${OpenCV2_ROOT_INC_DIR}/opencv2/imgproc")
find_path(OpenCV2_CONTRIB_INCLUDE_DIR    	NAMES contrib.hpp		PATHS "${OpenCV2_ROOT_INC_DIR}/opencv2/contrib")
find_path(OpenCV2_HIGHGUI_INCLUDE_DIR    	NAMES highgui.hpp		PATHS "${OpenCV2_ROOT_INC_DIR}/opencv2/highgui")
find_path(OpenCV2_FLANN_INCLUDE_DIR      	NAMES flann.hpp			PATHS "${OpenCV2_ROOT_INC_DIR}/opencv2/flann")
find_path(OpenCV2_CALIB3D_INCLUDE_DIR    	NAMES calib3d.hpp		PATHS "${OpenCV2_ROOT_INC_DIR}/opencv2/calib3d")
find_path(OpenCV2_FEATURES2D_INCLUDE_DIR    NAMES features2d.hpp    PATHS "${OpenCV2_ROOT_INC_DIR}/opencv2/features2d")
find_path(OpenCV2_GPU_INCLUDE_DIR    		NAMES gpu.hpp      		PATHS "${OpenCV2_ROOT_INC_DIR}/opencv2/gpu")
find_path(OpenCV2_LEGACY_INCLUDE_DIR    	NAMES legacy.hpp		PATHS "${OpenCV2_ROOT_INC_DIR}/opencv2/legacy")
find_path(OpenCV2_ML_INCLUDE_DIR    		NAMES ml.hpp			PATHS "${OpenCV2_ROOT_INC_DIR}/opencv2/ml")
#find_path(OpenCV2_NONFREE_INCLUDE_DIR    	NAMES nonfree.hpp		PATHS "${OpenCV2_ROOT_INC_DIR}/opencv2/nonfree")
#find_path(OpenCV2_OBJDETECT_INCLUDE_DIR    	NAMES objdetect.hpp		PATHS "${OpenCV2_ROOT_INC_DIR}/opencv2/objdetect")
find_path(OpenCV2_PHOTO_INCLUDE_DIR    		NAMES photo.hpp			PATHS "${OpenCV2_ROOT_INC_DIR}/opencv2/photo")
find_path(OpenCV2_STITCHING_INCLUDE_DIR    	NAMES stitcher.hpp		PATHS "${OpenCV2_ROOT_INC_DIR}/opencv2/stitching")
find_path(OpenCV2_TS_INCLUDE_DIR    		NAMES ts.hpp			PATHS "${OpenCV2_ROOT_INC_DIR}/opencv2/ts")
find_path(OpenCV2_VIDEO_INCLUDE_DIR    		NAMES video.hpp			PATHS "${OpenCV2_ROOT_INC_DIR}/opencv2/video")
find_path(OpenCV2_VIDEOSTAB_INCLUDE_DIR    NAMES videostab.hpp		PATHS "${OpenCV2_ROOT_INC_DIR}/opencv2/videostab")

set(OpenCV2_INCLUDE_DIRS
    ${OpenCV2_ROOT_INC_DIR}
    ${OpenCV2_ROOT_INC_DIR}/opencv2
    ${OpenCV2_CORE_INCLUDE_DIR}
    ${OpenCV2_IMGPROC_INCLUDE_DIR}
    ${OpenCV2_CONTRIB_INCLUDE_DIR}
    ${OpenCV2_HIGHGUI_INCLUDE_DIR}
    ${OpenCV2_FLANN_INCLUDE_DIR}
    ${OpenCV2_CALIB3D_INCLUDE_DIR}
    ${OpenCV2_FEATURES2D_INCLUDE_DIR}
    ${OpenCV2_GPU_INCLUDE_DIR}
    ${OpenCV2_LEGACY_INCLUDE_DIR}
    ${OpenCV2_ML_INCLUDE_DIR}
 #   ${OpenCV2_NONFREE_INCLUDE_DIR}
    ${OpenCV2_OBJDETECT_INCLUDE_DIR}
    ${OpenCV2_PHOTO_INCLUDE_DIR}
    ${OpenCV2_STITCHING_INCLUDE_DIR}
    ${OpenCV2_TS_INCLUDE_DIR}
    ${OpenCV2_VIDEO_INCLUDE_DIR}
    ${OpenCV2_VIDEOSTAB_INCLUDE_DIR})

# absolute path to all libraries 
# set(OPENCV2_LIBRARY_SEARCH_PATHS "${OpenCV2_ROOT_DIR}/lib")

#--- Specify where DLL is searched for
#message(STATUS "OPENCV2_LIBRARY_SEARCH_PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS}")
list(APPEND OPENCV2_LIBRARY_SEARCH_PATHS $ENV{OpenCV_ROOT_DIR})
list(APPEND OPENCV2_LIBRARY_SEARCH_PATHS "C:/Developer/lib")
list(APPEND OPENCV2_LIBRARY_SEARCH_PATHS "/usr/local/lib")
list(APPEND OPENCV2_LIBRARY_SEARCH_PATHS "/opt/local/lib")
list(APPEND OPENCV2_LIBRARY_SEARCH_PATHS "/usr/lib")


#--- FIND RELEASE LIBRARIES
find_library(OpenCV2_CORE_LIBRARY_REL			NAMES opencv_core opencv_core230 opencv_core220 opencv_core2410 opencv_core243                     PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
find_library(OpenCV2_IMGPROC_LIBRARY_REL		NAMES opencv_imgproc opencv_imgproc230 opencv_imgproc220 opencv_imgproc2410 opencv_imgproc243      PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
find_library(OpenCV2_CONTRIB_LIBRARY_REL		NAMES opencv_contrib opencv_contrib230 opencv_contrib220 opencv_contrib2410 opencv_contrib243      PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
find_library(OpenCV2_HIGHGUI_LIBRARY_REL		NAMES opencv_highgui opencv_highgui230 opencv_highgui220 opencv_highgui2410 opencv_highgui243      PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
find_library(OpenCV2_FLANN_LIBRARY_REL    		NAMES opencv_flann opencv_flann230 opencv_flann220 opencv_flann2410 opencv_flann243             	PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
find_library(OpenCV2_CALIB3D_LIBRARY_REL    	NAMES opencv_calib3d opencv_calib3d230 opencv_calib3d220 opencv_calib3d2410 opencv_calib3d243		PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
find_library(OpenCV2_FEATURES2D_LIBRARY_REL		NAMES opencv_features2d opencv_features2d230 opencv_features2d220 opencv_features2d2410 opencv_features2d243		PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
find_library(OpenCV2_GPU_LIBRARY_REL			NAMES opencv_gpu opencv_gpu230 opencv_gpu220 opencv_gpu2410 opencv_gpu243		PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
find_library(OpenCV2_LEGACY_LIBRARY_REL			NAMES opencv_legacy opencv_legacy230 opencv_legacy220 opencv_legacy2410 opencv_legacy243		PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
find_library(OpenCV2_ML_LIBRARY_REL				NAMES opencv_ml opencv_ml230 opencv_ml220 opencv_ml2410 opencv_ml243		PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
#find_library(OpenCV2_NONFREE_LIBRARY_REL		NAMES opencv_nonfree opencv_nonfree230 opencv_nonfree220 opencv_nonfree2410 opencv_nonfree243		PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
find_library(OpenCV2_OBJDETECT_LIBRARY_REL		NAMES opencv_objdetect opencv_objdetect230 opencv_objdetect220 opencv_objdetect2410 opencv_objdetect243		PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
find_library(OpenCV2_PHOTO_LIBRARY_REL			NAMES opencv_photo opencv_photo230 opencv_photo220 opencv_photo2410 opencv_photo243		PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
find_library(OpenCV2_STITCHING_LIBRARY_REL		NAMES opencv_stitching opencv_stitching230 opencv_stitching220 opencv_stitching2410 opencv_stitching243		PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
find_library(OpenCV2_TS_LIBRARY_REL				NAMES opencv_ts opencv_ts230 opencv_ts220 opencv_ts2410 opencv_ts243		PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
find_library(OpenCV2_VIDEO_LIBRARY_REL			NAMES opencv_video opencv_video230 opencv_video220 opencv_video2410 opencv_video243		PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
find_library(OpenCV2_VIDEOSTAB_LIBRARY_REL		NAMES opencv_videostab opencv_videostab230 opencv_videostab220 opencv_videostab2410 opencv_videostab243		PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
list(APPEND OpenCV2_LIBRARIES_REL ${OpenCV2_CORE_LIBRARY_REL})
list(APPEND OpenCV2_LIBRARIES_REL ${OpenCV2_IMGPROC_LIBRARY_REL})
list(APPEND OpenCV2_LIBRARIES_REL ${OpenCV2_CONTRIB_LIBRARY_REL})
list(APPEND OpenCV2_LIBRARIES_REL ${OpenCV2_HIGHGUI_LIBRARY_REL})
list(APPEND OpenCV2_LIBRARIES_REL ${OpenCV2_FLANN_LIBRARY_REL})
list(APPEND OpenCV2_LIBRARIES_REL ${OpenCV2_CALIB3D_LIBRARY_REL})
list(APPEND OpenCV2_LIBRARIES_REL ${OpenCV2_FEATURES2D_LIBRARY_REL})
list(APPEND OpenCV2_LIBRARIES_REL ${OpenCV2_GPU_LIBRARY_REL})
list(APPEND OpenCV2_LIBRARIES_REL ${OpenCV2_LEGACY_LIBRARY_REL})
list(APPEND OpenCV2_LIBRARIES_REL ${OpenCV2_ML_LIBRARY_REL})
#list(APPEND OpenCV2_LIBRARIES_REL ${OpenCV2_NONFREE_LIBRARY_REL})
list(APPEND OpenCV2_LIBRARIES_REL ${OpenCV2_OBJDETECT_LIBRARY_REL})
list(APPEND OpenCV2_LIBRARIES_REL ${OpenCV2_PHOTO_LIBRARY_REL})
list(APPEND OpenCV2_LIBRARIES_REL ${OpenCV2_STITCHING_LIBRARY_REL})
list(APPEND OpenCV2_LIBRARIES_REL ${OpenCV2_TS_LIBRARY_REL})
list(APPEND OpenCV2_LIBRARIES_REL ${OpenCV2_VIDEO_LIBRARY_REL})
list(APPEND OpenCV2_LIBRARIES_REL ${OpenCV2_VIDEOSTAB_LIBRARY_REL})

#--- FIND DEBUG LIBRARIES
if(WIN32)
    find_library(OpenCV2_CORE_LIBRARY_DEB       NAMES opencv_cored opencv_core230d opencv_core220d opencv_core2410d opencv_core243d                    PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
    find_library(OpenCV2_IMGPROC_LIBRARY_DEB    NAMES opencv_imgprocd opencv_imgproc230d opencv_imgproc220d opencv_imgproc2410d opencv_imgproc243d     PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
    find_library(OpenCV2_CONTRIB_LIBRARY_DEB    NAMES opencv_contribd opencv_contrib230d opencv_contrib220d opencv_contrib2410d opencv_contrib243d     PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
    find_library(OpenCV2_HIGHGUI_LIBRARY_DEB    NAMES opencv_highguid opencv_highgui230d opencv_highgui220d opencv_highgui2410d opencv_highgui243d     PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
    find_library(OpenCV2_FLANN_LIBRARY_DEB    	NAMES opencv_flannd opencv_flann230d opencv_flann220d opencv_flann2410d opencv_flann243d         		PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
    find_library(OpenCV2_CALIB3D_LIBRARY_DEB    NAMES opencv_calib3dd opencv_calib3d230d opencv_calib3d220d opencv_calib3d2410d opencv_calib3d243d     PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
	find_library(OpenCV2_FEATURES2D_LIBRARY_DEB		NAMES opencv_features2dd opencv_features2d230d opencv_features2d220d opencv_features2d2410d opencv_features2d243d		PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
	find_library(OpenCV2_GPU_LIBRARY_DEB			NAMES opencv_gpud opencv_gpu230d opencv_gpu220d opencv_gpu2410d opencv_gpu243d		PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
	find_library(OpenCV2_LEGACY_LIBRARY_DEB			NAMES opencv_legacyd opencv_legacy230d opencv_legacy220d opencv_legacy2410d opencv_legacy243d		PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
	find_library(OpenCV2_ML_LIBRARY_DEB				NAMES opencv_mld opencv_ml230d opencv_ml220d opencv_ml2410d opencv_ml243d		PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
#	find_library(OpenCV2_NONFREE_LIBRARY_DEB		NAMES opencv_nonfreed opencv_nonfree230d opencv_nonfree220d opencv_nonfree2410d opencv_nonfree243d		PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
	find_library(OpenCV2_OBJDETECT_LIBRARY_DEB		NAMES opencv_objdetectd opencv_objdetect230d opencv_objdetect220d opencv_objdetect2410d opencv_objdetect243d		PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
	find_library(OpenCV2_PHOTO_LIBRARY_DEB			NAMES opencv_photod opencv_photo230d opencv_photo220d opencv_photo2410d opencv_photo243d		PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
	find_library(OpenCV2_STITCHING_LIBRARY_DEB		NAMES opencv_stitchingd opencv_stitching230d opencv_stitching220d opencv_stitching2410d opencv_stitching243d		PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
	find_library(OpenCV2_TS_LIBRARY_DEB				NAMES opencv_tsd opencv_ts230d opencv_ts220d opencv_ts2410d opencv_ts243d		PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
	find_library(OpenCV2_VIDEO_LIBRARY_DEB			NAMES opencv_videod opencv_video230d opencv_video220d opencv_video2410d opencv_video243d		PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
	find_library(OpenCV2_VIDEOSTAB_LIBRARY_DEB		NAMES opencv_videostabd opencv_videostab230d opencv_videostab220d opencv_videostab2410d opencv_videostab243d		PATHS ${OPENCV2_LIBRARY_SEARCH_PATHS})
	list(APPEND OpenCV2_LIBRARIES_DEB ${OpenCV2_CORE_LIBRARY_DEB})
    list(APPEND OpenCV2_LIBRARIES_DEB ${OpenCV2_IMGPROC_LIBRARY_DEB})
    list(APPEND OpenCV2_LIBRARIES_DEB ${OpenCV2_CONTRIB_LIBRARY_DEB})
    list(APPEND OpenCV2_LIBRARIES_DEB ${OpenCV2_HIGHGUI_LIBRARY_DEB})
    list(APPEND OpenCV2_LIBRARIES_DEB ${OpenCV2_FLANN_LIBRARY_DEB})
    list(APPEND OpenCV2_LIBRARIES_DEB ${OpenCV2_CALIB3D_LIBRARY_DEB})
	list(APPEND OpenCV2_LIBRARIES_DEB ${OpenCV2_FEATURES2D_LIBRARY_DEB})
	list(APPEND OpenCV2_LIBRARIES_DEB ${OpenCV2_GPU_LIBRARY_DEB})
	list(APPEND OpenCV2_LIBRARIES_DEB ${OpenCV2_LEGACY_LIBRARY_DEB})
	list(APPEND OpenCV2_LIBRARIES_DEB ${OpenCV2_ML_LIBRARY_DEB})
#	list(APPEND OpenCV2_LIBRARIES_DEB ${OpenCV2_NONFREE_LIBRARY_DEB})
	list(APPEND OpenCV2_LIBRARIES_DEB ${OpenCV2_OBJDETECT_LIBRARY_DEB})
	list(APPEND OpenCV2_LIBRARIES_DEB ${OpenCV2_PHOTO_LIBRARY_DEB})
	list(APPEND OpenCV2_LIBRARIES_DEB ${OpenCV2_STITCHING_LIBRARY_DEB})
	list(APPEND OpenCV2_LIBRARIES_DEB ${OpenCV2_TS_LIBRARY_DEB})
	list(APPEND OpenCV2_LIBRARIES_DEB ${OpenCV2_VIDEO_LIBRARY_DEB})
	list(APPEND OpenCV2_LIBRARIES_DEB ${OpenCV2_VIDEOSTAB_LIBRARY_DEB})
endif()

#--- Setup cross-config libraries
set(OpenCV2_LIBRARIES "")
if(WIN32)
    list(APPEND OpenCV2_LIBRARIES optimized ${OpenCV2_CORE_LIBRARY_REL}    debug ${OpenCV2_CORE_LIBRARY_DEB})
    list(APPEND OpenCV2_LIBRARIES optimized ${OpenCV2_IMGPROC_LIBRARY_REL} debug ${OpenCV2_IMGPROC_LIBRARY_DEB})
    list(APPEND OpenCV2_LIBRARIES optimized ${OpenCV2_CONTRIB_LIBRARY_REL} debug ${OpenCV2_CONTRIB_LIBRARY_DEB})
    list(APPEND OpenCV2_LIBRARIES optimized ${OpenCV2_HIGHGUI_LIBRARY_REL} debug ${OpenCV2_HIGHGUI_LIBRARY_DEB})
    list(APPEND OpenCV2_LIBRARIES optimized ${OpenCV2_FLANN_LIBRARY_REL}   debug ${OpenCV2_FLANN_LIBRARY_DEB})
    list(APPEND OpenCV2_LIBRARIES optimized ${OpenCV2_CALIB3D_LIBRARY_REL} debug ${OpenCV2_CALIB3D_LIBRARY_DEB})
    list(APPEND OpenCV2_LIBRARIES optimized ${OpenCV2_FEATURES2D_LIBRARY_REL} debug ${OpenCV2_FEATURES2D_LIBRARY_DEB})
    list(APPEND OpenCV2_LIBRARIES optimized ${OpenCV2_GPU_LIBRARY_REL} debug ${OpenCV2_GPU_LIBRARY_DEB})
    list(APPEND OpenCV2_LIBRARIES optimized ${OpenCV2_LEGACY_LIBRARY_REL} debug ${OpenCV2_LEGACY_LIBRARY_DEB})
    list(APPEND OpenCV2_LIBRARIES optimized ${OpenCV2_ML_LIBRARY_REL} debug ${OpenCV2_ML_LIBRARY_DEB})
#    list(APPEND OpenCV2_LIBRARIES optimized ${OpenCV2_NONFREE_LIBRARY_REL} debug ${OpenCV2_NONFREE_LIBRARY_DEB})
    list(APPEND OpenCV2_LIBRARIES optimized ${OpenCV2_OBJDETECT_LIBRARY_REL} debug ${OpenCV2_OBJDETECT_LIBRARY_DEB})
    list(APPEND OpenCV2_LIBRARIES optimized ${OpenCV2_PHOTO_LIBRARY_REL} debug ${OpenCV2_PHOTO_LIBRARY_DEB})
    list(APPEND OpenCV2_LIBRARIES optimized ${OpenCV2_STITCHING_LIBRARY_REL} debug ${OpenCV2_STITCHING_LIBRARY_DEB})
    list(APPEND OpenCV2_LIBRARIES optimized ${OpenCV2_TS_LIBRARY_REL} debug ${OpenCV2_TS_LIBRARY_DEB})
    list(APPEND OpenCV2_LIBRARIES optimized ${OpenCV2_VIDEO_LIBRARY_REL} debug ${OpenCV2_VIDEO_LIBRARY_DEB})
    list(APPEND OpenCV2_LIBRARIES optimized ${OpenCV2_VIDEOSTAB_LIBRARY_REL} debug ${OpenCV2_VIDEOSTAB_LIBRARY_DEB})
else()
    list(APPEND OpenCV2_LIBRARIES ${OpenCV2_CORE_LIBRARY_REL})
    list(APPEND OpenCV2_LIBRARIES ${OpenCV2_IMGPROC_LIBRARY_REL})
    list(APPEND OpenCV2_LIBRARIES ${OpenCV2_CONTRIB_LIBRARY_REL})
    list(APPEND OpenCV2_LIBRARIES ${OpenCV2_HIGHGUI_LIBRARY_REL})
    list(APPEND OpenCV2_LIBRARIES ${OpenCV2_FLANN_LIBRARY_REL})
    list(APPEND OpenCV2_LIBRARIES ${OpenCV2_CALIB3D_LIBRARY_REL})
    list(APPEND OpenCV2_LIBRARIES ${OpenCV2_FEATURES2D_LIBRARY_REL})
    list(APPEND OpenCV2_LIBRARIES ${OpenCV2_GPU_LIBRARY_REL})
    list(APPEND OpenCV2_LIBRARIES ${OpenCV2_LEGACY_LIBRARY_REL})
    list(APPEND OpenCV2_LIBRARIES ${OpenCV2_ML_LIBRARY_REL})
 #   list(APPEND OpenCV2_LIBRARIES ${OpenCV2_NONFREE_LIBRARY_REL})
    list(APPEND OpenCV2_LIBRARIES ${OpenCV2_OBJDETECT_LIBRARY_REL})
    list(APPEND OpenCV2_LIBRARIES ${OpenCV2_PHOTO_LIBRARY_REL})
    list(APPEND OpenCV2_LIBRARIES ${OpenCV2_STITCHING_LIBRARY_REL})
    list(APPEND OpenCV2_LIBRARIES ${OpenCV2_TS_LIBRARY_REL})
    list(APPEND OpenCV2_LIBRARIES ${OpenCV2_VIDEO_LIBRARY_REL})
    list(APPEND OpenCV2_LIBRARIES ${OpenCV2_VIDEOSTAB_LIBRARY_REL})
endif()

#--- Verifies everything (include) was found
set(OpenCV2_FOUND ON)
FOREACH(NAME ${OpenCV2_INCLUDE_DIRS})
    IF(NOT EXISTS ${NAME})
        message(WARNING "Could not find: ${NAME}")
        set(OpenCV2_FOUND OFF)
    endif(NOT EXISTS ${NAME})
ENDFOREACH(NAME)

#--- Verifies everything (release lib) was found
FOREACH(NAME ${OpenCV2_LIBRARIES_REL})
    IF(NOT EXISTS ${NAME})
        message(WARNING "Could not find: ${NAME}")
        set(OpenCV2_FOUND OFF)
    endif(NOT EXISTS ${NAME})
 ENDFOREACH()

#--- Verifies everything (debug lib) was found
FOREACH(NAME ${OpenCV2_LIBRARIES_DEB})
    IF(NOT EXISTS ${NAME})
        message(WARNING "Could not find: ${NAME}")
        set(OpenCV2_FOUND OFF)
    endif(NOT EXISTS ${NAME})
ENDFOREACH()

#--- Display help message
IF(NOT OpenCV2_FOUND)
    IF(OpenCV2_FIND_REQUIRED)
        MESSAGE(FATAL_ERROR "OpenCV 2 not found.")
    else()
        MESSAGE(STATUS "OpenCV 2 not found.")
    endif()
endif()
