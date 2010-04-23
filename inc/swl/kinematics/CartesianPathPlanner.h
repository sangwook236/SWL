#if !defined(__SWL_KINEMATICS__CARTESIAN_PATH_PLANNER__H_)
#define __SWL_KINEMATICS__CARTESIAN_PATH_PLANNER__H_ 1


#include "swl/kinematics/PathPlanner.h"
#include "swl/math/Vector.h"
#include "swl/math/Quaternion.h"


namespace swl {

#if defined(_MSC_VER)
#pragma warning(disable:4231)
SWL_KINEMATICS_EXPORT_TEMPLATE template class SWL_KINEMATICS_API Vector3<double>;
SWL_KINEMATICS_EXPORT_TEMPLATE template class SWL_KINEMATICS_API Quaternion<double>;
#endif


//--------------------------------------------------------------------------------
// class CartesianPathPlanner

class SWL_KINEMATICS_API CartesianPathPlanner: public PathPlanner
{
public:
	typedef PathPlanner base_type;

public:
	CartesianPathPlanner(KinematicsBase &kinem);
	explicit CartesianPathPlanner(const CartesianPathPlanner &rhs);
	virtual ~CartesianPathPlanner();

private:
	CartesianPathPlanner & operator=(const CartesianPathPlanner &rhs);

public:
	///
	/*virtual*/ void reset();

protected:
	/// start & end position of a path
	Vector3<double> startPt_, endPt_;

	/// start & end orientation of a path
	Quaternion<double> startQuat_, endQuat_;
};

}  // namespace swl


#endif  // __SWL_KINEMATICS__CARTESIAN_PATH_PLANNER__H_
