#if !defined(__SWL_KINEMATICS__CURVE_PATH_PLANNER__H_)
#define __SWL_KINEMATICS__CURVE_PATH_PLANNER__H_ 1


#include "swl/kinematics/CartesianPathPlanner.h"


namespace swl {

//--------------------------------------------------------------------------------
// class CurvePathPlanner

class SWL_KINEMATICS_API CurvePathPlanner: public CartesianPathPlanner
{
public:
	typedef CartesianPathPlanner base_type;

public:
	enum ECurve { CURVE_PTP = 0, CURVE_BEZIER, CURVE_NURBS };

private:
	explicit CurvePathPlanner(const CurvePathPlanner &rhs);
	CurvePathPlanner & operator=(const CurvePathPlanner &rhs);

public:
	CurvePathPlanner(KinematicsBase &kinem, const ECurve eCurve);
	virtual ~CurvePathPlanner();

public:
	///
	/*virtual*/ bool plan();
	/*virtual*/ void reset();

	/// in joint coordinates
	/*virtual*/ bool getNextPose(coords_type &aPose, const coords_type *refPose = NULL);

	/// in joint coordinates
	/*virtual*/ void addViaPose(const coords_type &aViaPose);
	/*virtual*/ void addViaPose(pose_ctr::const_iterator citFirstPose, pose_ctr::const_iterator citLastPose);

protected:
	///
	pose_ctr poseCtr_;

	/// curve type
	ECurve curveType_;
};

}  // namespace swl


#endif  // __SWL_KINEMATICS__CURVE_PATH_PLANNER__H_
