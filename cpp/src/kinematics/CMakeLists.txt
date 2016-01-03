set(TARGET swl_kinematics)

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
set(LIBS
	swl_math
	swl_base
)

add_definitions(-DSWL_KINEMATICS_EXPORT)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	add_compile_options(-fPIC)
endif(CMAKE_CXX_COMPILER_ID MATCHES GNU)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
