#if !defined(__SWL_KINEMATICS__JOINT_PATH_PLANNER__H_)
#define __SWL_KINEMATICS__JOINT_PATH_PLANNER__H_ 1


#include "swl/kinematics/PathPlanner.h"


namespace swl {

//--------------------------------------------------------------------------------
// class JointPathPlanner

class SWL_KINEMATICS_API JointPathPlanner: public PathPlanner
{
public:
	typedef PathPlanner base_type;

public:
	JointPathPlanner(KinematicsBase &kinem);
	virtual ~JointPathPlanner();

private:
	explicit JointPathPlanner(const JointPathPlanner &rhs);
	JointPathPlanner & operator=(const JointPathPlanner &rhs);

public:
	///
	/*virtual*/ bool plan();

	/// in joint coordinates
	/*virtual*/ bool getNextPose(coords_type &aPose, const coords_type *refPose = 0L);

	/// in joint coordinates
	/*virtual*/ void addViaPose(const coords_type & /*aViaPose*/)  {}
	/*virtual*/ void addViaPose(pose_ctr::const_iterator /*citFirstPose*/, pose_ctr::const_iterator /*citLastPose*/)  {}
};

}  // namespace swl


#endif  // __SWL_KINEMATICS__JOINT_PATH_PLANNER__H_
