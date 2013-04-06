#include "swl/Config.h"
#include "swl/kinematics/Kinematics.h"
#include "swl/base/LogException.h"
#include <iterator>
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------------
// class KinematicsBase

KinematicsBase::KinematicsBase()
: //base_type(),
  deviceType_(DEVICE_UNDEFINED)
{
}

KinematicsBase::~KinematicsBase()
{
}

bool KinematicsBase::checkJointLimit(const size_t jointId, const double jointValue)
{
    if (jointId < 0 || jointId >= getDOF()) return false;
	return getJointParam(jointId).getLowerLimit() <= jointValue && jointValue <= getJointParam(jointId).getUpperLimit();
}


//--------------------------------------------------------------------------------
//  class Kinematics

Kinematics::Kinematics()
: base_type(),
  screwAxisCtr_()
{
}

Kinematics::~Kinematics()
{
}

JointParam & Kinematics::getJointParam(const size_t jointId) const
{
	// FIXME [add] >>
	throw std::logic_error("not yet implemented");
}

void Kinematics::addScrewAxis(const ScrewAxis &screwAxis)
{  screwAxisCtr_.push_back(screwAxis);  }

ScrewAxis & Kinematics::getScrewAxis(const size_t jointId)
{
    if (jointId < 0 || jointId >= getDOF())
		throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
    return screwAxisCtr_[jointId];
}

const ScrewAxis & Kinematics::getScrewAxis(const size_t jointId) const
{
    if (jointId < 0 || jointId >= getDOF())
		throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
    return screwAxisCtr_[jointId];
}

void Kinematics::removeScrewAxis(const size_t jointId)
{
    if (jointId < 0 || jointId >= getDOF()) return;
	screw_ctr::iterator itScrew = screwAxisCtr_.begin();
	std::advance(itScrew, jointId);
	screwAxisCtr_.erase(itScrew);
}

}  // namespace swl
