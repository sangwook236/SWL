#if !defined(__SWL_KINEMATICS__LINE_PATH_PLANNER__H_)
#define __SWL_KINEMATICS__LINE_PATH_PLANNER__H_ 1


#include "swl/kinematics/CartesianPathPlanner.h"


namespace swl {

//--------------------------------------------------------------------------------
// class LinePathPlanner

class SWL_KINEMATICS_API LinePathPlanner: public CartesianPathPlanner
{
public:
	typedef CartesianPathPlanner base_type;

public:
	LinePathPlanner(KinematicsBase &kinem);
	virtual ~LinePathPlanner();

private:
	explicit LinePathPlanner(const LinePathPlanner &rhs);
	LinePathPlanner& operator=(const LinePathPlanner &rhs);

public:
	///
	/*virtual*/ bool plan();

	///  in joint coordinates
	/*virtual*/ bool getNextPose(coords_type &aPose, const coords_type *refPose = NULL);

	///  in joint coordinates
	/*virtual*/ void addViaPose(const coords_type & /*aViaPose*/)  {}
	/*virtual*/ void addViaPose(CartesianPathPlanner::pose_ctr::const_iterator /*citFirstPose*/, CartesianPathPlanner::pose_ctr::const_iterator /*citLastPose*/)  {}
//@}
};

}  // namespace swl


#endif  // __SWL_KINEMATICS__LINE_PATH_PLANNER__H_
