#include "swl/Config.h"
#include "swl/kinematics/PathPlanner.h"
#include "swl/kinematics/Kinematics.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------------
// class PathPlanner

PathPlanner::PathPlanner(KinematicsBase &kinem)
: //base_type(),
  kinematics_(kinem), dof_(kinem.getDOF()),
  initPose_(dof_), finalPose_(dof_), currPose_(dof_),
  velocityRatio_(1.0), samplingTime_(100u), currStep_(0u), maxStep_(0u)
{
}

PathPlanner::PathPlanner(const PathPlanner &rhs)
: //base_type(rhs),
  kinematics_(rhs.kinematics_), dof_(rhs.dof_),
  initPose_(rhs.initPose_), finalPose_(rhs.finalPose_), currPose_(rhs.currPose_),
  velocityRatio_(rhs.velocityRatio_), samplingTime_(rhs.samplingTime_), currStep_(rhs.currStep_), maxStep_(rhs.maxStep_)
{
}

PathPlanner::~PathPlanner()
{
}
/*
PathPlanner & PathPlanner::operator=(const PathPlanner &rhs)
{
	if (this == &rhs) return *this;
	//static_cast<base_type &>(*this) = rhs;

	kinematics_ = rhs.kinematics_;
	dof_ = rhs.dof_;

	initPose_.assign(rhs.initPose_.begin(), rhs.initPose_.end());
	finalPose_.assign(rhs.finalPose_.begin(), rhs.finalPose_.end());
	currPose_.assign(rhs.currPose_.begin(), rhs.currPose_.end());

	velocityRatio_ = rhs.velocityRatio_;
	samplingTime_ = rhs.samplingTime_;
	currStep_ = rhs.currStep_;
	maxStep_ = rhs.maxStep_;
	return *this;
}
*/
void PathPlanner::reset()
{
	//kinematics_ = ;
	//dof_ = 0u;
	initPose_.clear();  initPose_.reserve(dof_);
	finalPose_.clear();  finalPose_.reserve(dof_);
	currPose_.clear();  currPose_.reserve(dof_);
	velocityRatio_ = 1.0;
	samplingTime_ = 100u;
	resetStep();
}

const PathPlanner::coords_type & PathPlanner::getCurrPose() const
{  return currPose_;  }

void PathPlanner::setInitPose(const PathPlanner::coords_type &aInitPose)
{  initPose_.assign(aInitPose.begin(), aInitPose.end());  }

void PathPlanner::setFinalPose(const PathPlanner::coords_type &aFinalPose)
{  finalPose_.assign(aFinalPose.begin(), aFinalPose.end());  }

bool PathPlanner::isDone() const
{
	if (!maxStep_) return true;
	else if (currStep_ != maxStep_) return false;

	double dEPS = swl::MathConstant::EPS;
	for (size_t i = 0; i< dof_; ++i)
	{
		if (fabs(currPose_[i] - finalPose_[i]) > dEPS) 
			return false;
	}
	return true;
}

}  // namespace swl
