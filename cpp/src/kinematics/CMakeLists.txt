if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_kinematics_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_kinematics)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	ArcPathPlanner.cpp
	ArticulatedKinematics.cpp
	CartesianKinematics.cpp
	CartesianPathPlanner.cpp
	CurvePathPlanner.cpp
	DHParam.cpp
	Joint.cpp
	JointParam.cpp
	JointPathPlanner.cpp
	Kinematics.cpp
	LinePathPlanner.cpp
	Link.cpp
	ParallelogramLinkKinematics.cpp
	PathBlender.cpp
	PathPlanner.cpp
	PumaKinematics.cpp
	RobotKinematics.cpp
	ScaraKinematics.cpp
	ScrewAxis.cpp
	StanfordArmKinematics.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_math_d
		swl_base_d
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_math
		swl_base
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_definitions(-DSWL_KINEMATICS_EXPORT)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
