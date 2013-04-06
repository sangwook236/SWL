#include "swl/Config.h"
#include "swl/kinematics/CartesianPathPlanner.h"
#include "swl/kinematics/Kinematics.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------------
// class CartesianPathPlanner

CartesianPathPlanner::CartesianPathPlanner(KinematicsBase &kinem)
: base_type(kinem),
  startPt_(), endPt_(), startQuat_(), endQuat_()
{
}

CartesianPathPlanner::CartesianPathPlanner(const CartesianPathPlanner &rhs)
: base_type(rhs),
  startPt_(rhs.startPt_), endPt_(rhs.endPt_), startQuat_(rhs.startQuat_), endQuat_(rhs.endQuat_)
{
}

CartesianPathPlanner::~CartesianPathPlanner()
{
}

void CartesianPathPlanner::reset()
{
	base_type::reset();

	startPt_ = endPt_ = Vector3<double>();
	startQuat_ = endQuat_ = Quaternion<double>();
}

}  // namespace swl
