#include "swl/Config.h"
#include "swl/kinematics/LinePathPlanner.h"
#include "swl/kinematics/Kinematics.h"
#include "swl/math/Rotation.h"
#include <algorithm>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------------
// class LinePathPlanner

LinePathPlanner::LinePathPlanner(KinematicsBase &kinem)
: base_type(kinem)
{
}

LinePathPlanner::LinePathPlanner(const LinePathPlanner &rhs)
: base_type(rhs)
{
}

LinePathPlanner::~LinePathPlanner()
{
}

bool LinePathPlanner::getNextPose(LinePathPlanner::coords_type &aPose, const LinePathPlanner::coords_type *refPose /*= NULL*/)
{
	// TODO [add] >>
	return true;
}

bool LinePathPlanner::plan()
{
	currStep_ = 0u;
	currPose_.assign(initPose_.begin(), initPose_.end());

	//
	coords_type aInitCartesian(dof_), aFinalCartesian(dof_);
	if (!kinematics_.solveForward(initPose_, aInitCartesian) ||
		!kinematics_.solveForward(finalPose_, aFinalCartesian))
		return false;

	coords_type aInitSpatial(kinematics_.cartesianToSpatial(aInitCartesian)), aFinalSpatial(kinematics_.cartesianToSpatial(aFinalCartesian));

	memcpy(&startPt_.x(), &aInitSpatial[0], 3 * sizeof(coords_type::value_type));
	memcpy(&endPt_.x(), &aFinalSpatial[0], 3 * sizeof(coords_type::value_type));

	startQuat_ = Quaternion<double>::toQuaternion(Rotation::rotate(kinematics_.getRotOrder(), TVector3<double>(aInitSpatial[3], aInitSpatial[4], aInitSpatial[5])));
	endQuat_ = Quaternion<double>::toQuaternion(Rotation::rotate(kinematics_.getRotOrder(), TVector3<double>(aFinalSpatial[3], aFinalSpatial[4], aFinalSpatial[5])));

	// calculate required time
	double dTmp;
	{
		std::vector<double> vtrTime(dof_);
		for (size_t i = 0 ; i < dof_ ; ++i)
		{
			//vtrTime[i] = fabs((finalPose_[i] - initPose_[i]) / (kinematics_.getAxisParam(i).GetMaxJointSpeed() * velocityRatio_));
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
