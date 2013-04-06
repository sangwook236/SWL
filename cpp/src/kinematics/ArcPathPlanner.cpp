#include "swl/Config.h"
#include "swl/kinematics/ArcPathPlanner.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------------
// class ArcPathPlanner

ArcPathPlanner::ArcPathPlanner(KinematicsBase &kinem, const bool isCircular /*= false*/)
: base_type(kinem),
  viaPose_(dof_), isCircular_(isCircular),
  center_(), radius_(0.0), angle_(0.0), normal_()
{
}

ArcPathPlanner::ArcPathPlanner(const ArcPathPlanner &rhs)
: base_type(rhs),
  viaPose_(rhs.viaPose_), isCircular_(rhs.isCircular_),
  center_(rhs.center_), radius_(rhs.radius_), angle_(rhs.angle_), normal_(rhs.normal_)
{
}

ArcPathPlanner::~ArcPathPlanner()
{
}

void ArcPathPlanner::reset()
{
	base_type::reset();

	viaPose_.clear();  viaPose_.reserve(dof_);
	//isCircular_ = false;
	center_ = Vector3<double>();
	radius_ = 0.0;
	angle_ = 0.0;
	normal_ = Vector3<double>();
}

bool ArcPathPlanner::getNextPose(ArcPathPlanner::coords_type &aPose, const ArcPathPlanner::coords_type *refPose /*= NULL*/)
{
	// TODO [add] >>
	return true;
}

void ArcPathPlanner::addViaPose(const ArcPathPlanner::coords_type &aViaPose)
{  viaPose_.assign(aViaPose.begin(), aViaPose.end());  }

bool ArcPathPlanner::plan()
{
	currStep_ = 0u;
	currPose_.assign(initPose_.begin(), initPose_.end());

	// TODO [add] >>
	return true;
}

}  // namespace swl
