#include "swl/Config.h"
#include "swl/kinematics/JointPathPlanner.h"
#include "swl/kinematics/Kinematics.h"
#include <algorithm>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------------
// class JointPathPlanner

JointPathPlanner::JointPathPlanner(KinematicsBase &kinem)
: base_type(kinem)
{
}

JointPathPlanner::JointPathPlanner(const JointPathPlanner &rhs)
: base_type(rhs)
{
}

JointPathPlanner::~JointPathPlanner()
{
}

bool JointPathPlanner::getNextPose(JointPathPlanner::coords_type &aPose, const JointPathPlanner::coords_type * /*refPose = NULL*/)
{
	bool isUpdated = true;
	if (++currStep_ > maxStep_)
	{
		currStep_ = maxStep_;
		isUpdated = false;
	}

	if (!maxStep_)
		aPose.assign(initPose_.begin(), initPose_.end());
	else if (currStep_ == maxStep_)
	{
		aPose.assign(finalPose_.begin(), finalPose_.end());
		if (isUpdated)
			currPose_.assign(finalPose_.begin(), finalPose_.end());
	}
	else
	{
		for (size_t i = 0; i < dof_; ++i)
			currPose_[i] = initPose_[i] + (finalPose_[i] - initPose_[i]) * double(currStep_) / double(maxStep_);
		aPose.assign(currPose_.begin(), currPose_.end());
	}
	return true;
}

bool JointPathPlanner::plan()
{
	currStep_ = 0u;
	currPose_.assign(initPose_.begin(), initPose_.end());

	// calculate required time
	double dTmp;
	{
		std::vector<double> vtrTime(dof_);
		for (size_t i = 0; i < dof_; ++i)
		{
			//vtrTime[i] = fabs((finalPose_[i] - initPose_[i]) / (kinematics_.GetAxisParam(i).GetMaxJointSpeed() * velocityRatio_));
			dTmp = (finalPose_[i] - initPose_[i]) / (kinematics_.getJointParam(i).getMaxJointSpeed() * velocityRatio_);
			vtrTime[i] = dTmp > 0.0 ? dTmp : -dTmp;
		}
		dTmp = *std::max_element(vtrTime.begin(), vtrTime.end()) * 1000.0 / samplingTime_;
	}

	// calculate max. step
	//maxStep_ = (unsigned int)ceil(dTmp);
	maxStep_ = (unsigned int)dTmp;
	if (dTmp > maxStep_) ++maxStep_;
	return true;
}

}  // namespace swl
