#include "swl/Config.h"
#include "swl/kinematics/RobotKinematics.h"
#include "swl/base/LogException.h"
#include "swl/math/MathConstant.h"
#include <iterator>
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------------
// class RobotKinematics

RobotKinematics::RobotKinematics()
: base_type(),
  dhParamCtr_()
{
}

RobotKinematics::~RobotKinematics()
{
}

JointParam & RobotKinematics::getJointParam(const size_t jointId) const
{
	// FIXME [add] >>
	throw std::logic_error("Not yet implemented");
}

void RobotKinematics::addDHParam(const DHParam &dhParam)
{  dhParamCtr_.push_back(dhParam);  }

DHParam & RobotKinematics::getDHParam(const size_t jointId)
{
    if (jointId < 0 || jointId >= getDOF())
		throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
    return dhParamCtr_[jointId];
}

const DHParam & RobotKinematics::getDHParam(const size_t jointId) const
{
    if (jointId < 0 || jointId >= getDOF())
		throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
    return dhParamCtr_[jointId];
}

void RobotKinematics::removeDHParam(const size_t jointId)
{
    if (jointId < 0 || jointId >= getDOF()) return;
	dh_ctr::iterator itDH = dhParamCtr_.begin();
	std::advance(itDH, jointId);
	dhParamCtr_.erase(itDH);
}

bool RobotKinematics::isSingular(const RobotKinematics::coords_type &poseCoords, const bool isJointSpace /*= true*/)
// if isJointSpace == true,
//  poseCoords: joint values in joint coordinates, [rad]
// if isJointSpace == false,
//  poseCoords: cartesian values in cartesian coordinates, [m ; rad]
// if singular, return true
{
	const double tolerance = MathConstant::EPS;
	const double det = calcJacobianDeterminant(poseCoords, isJointSpace);

	if (-tolerance <= det && det <= tolerance)
		throw LogException(LogException::L_INFO, "the current robot's pose Is nearly singular", __FILE__, __LINE__, __FUNCTION__);
	else return false;
}

bool RobotKinematics::isReachable(const RobotKinematics::coords_type &poseCoords, const bool isJointSpace /*= true*/)
// if isJointSpace == true,
//  poseCoords: joint values in joint coordinates, [rad]
// if isJointSpace == false,
//  poseCoords: cartesian values in cartesian coordinates, [m ; rad]
// if reachable, return true
{
	if (isJointSpace)  // joint coordinates
	{
		for (unsigned int i = 0 ; i < getDOF() ; ++i)
			if (!checkJointLimit(i, poseCoords[i])) return false;
	}
	else  // cartesian coordinates
	{
		coords_type vJoint;
		return solveInverse(poseCoords, vJoint);
	}

	return true;
}

}  // namespace swl
