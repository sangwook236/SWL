#if !defined(__SWL_KINEMATICS__ARC_PATH_PLANNER__H_)
#define __SWL_KINEMATICS__ARC_PATH_PLANNER__H_ 1


#include "swl/kinematics/CartesianPathPlanner.h"


namespace swl {

//--------------------------------------------------------------------------------
// class ArcPathPlanner

class SWL_KINEMATICS_API ArcPathPlanner: public CartesianPathPlanner
{
public:
	typedef CartesianPathPlanner base_type;

public:
	ArcPathPlanner(KinematicsBase &kinem, const bool isCircular = false);
	virtual ~ArcPathPlanner();

private:
	explicit ArcPathPlanner(const ArcPathPlanner &rhs);
	ArcPathPlanner & operator=(const ArcPathPlanner &rhs);

public:
	///
	/*virtual*/ bool plan();
	/*virtual*/ void reset();

	/// in joint coordinates
	/*virtual*/ bool getNextPose(coords_type &aPose, const coords_type *refPose = NULL);

	/// in joint coordinates
	/*virtual*/ void addViaPose(const coords_type &aViaPose);
	/*virtual*/ void addViaPose(CartesianPathPlanner::pose_ctr::const_iterator /*citFirstPose*/, CartesianPathPlanner::pose_ctr::const_iterator /*citLastPose*/)  {}
//@}

protected:
	///
	coords_type viaPose_;

	///
	bool isCircular_;

	/// center of an arc
	Vector3<double> center_;
	/// radius and angle of an arc
	double radius_, angle_;
	/// normal vector of a plane including an arc
	Vector3<double> normal_;
};

}  // namespace swl


#endif  // __SWL_KINEMATICS__ARC_PATH_PLANNER__H_
