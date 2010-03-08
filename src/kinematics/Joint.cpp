#include "swl/Config.h"
#include "swl/kinematics/Joint.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// class Joint

Joint::Joint()
: //base_type(),
  jointParam_(JointParam::JOINT_UNKNOWN, 0.0, 0.0, 0.0)
{
}

Joint::Joint(const Joint &rhs)
: //base_type(rhs),
  jointParam_(rhs.jointParam_)
{
}

Joint::~Joint()
{
}

Joint & Joint::operator=(const Joint &rhs)
{
	if (this == &rhs) return *this;
	//static_cast<Joint::base_type &>(*this) = rhs;
	jointParam_ = rhs.jointParam_;
	return *this;
}

}  // namespace swl
