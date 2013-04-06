#include "swl/Config.h"
#include "swl/kinematics/JointParam.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// class JointParam

JointParam::JointParam()
: //base_type(),
  jointType_(JOINT_UNDEFINED), lowerLimit_(0.0), upperLimit_(0.0),
  maxJointSpeed_(0.0)
{}

/*
JointParam::JointParam(const JointParam::EJoint jointType, const double lowerLimit, const double upperLimit, const double maxJointSpeed, const double reductionRatio, const unsinged int encoderResolution)
: //base_type(),
  jointType_(jointType), lowerLimit_(lowerLimit), upperLimit_(upperLimit),
  maxJointSpeed_(maxJointSpeed),
  reductionRatio_(reductionRatio), encoderResolution_(encoderResolution)
*/
JointParam::JointParam(const JointParam::EJoint jointType, const double lowerLimit, const double upperLimit, const double maxJointSpeed)
: //base_type(),
  jointType_(jointType), lowerLimit_(lowerLimit), upperLimit_(upperLimit),
  maxJointSpeed_(maxJointSpeed)
{}

JointParam::JointParam(const JointParam &rhs)
: //base_type(rhs),
  jointType_(rhs.jointType_), lowerLimit_(rhs.lowerLimit_), upperLimit_(rhs.upperLimit_),
/*
  maxJointSpeed_(rhs.maxJointSpeed_),
  reductionRatio_(rhs.reductionRatio_), encoderResolution_(rhs.encoderResolution_)
*/
  maxJointSpeed_(rhs.maxJointSpeed_)
{}

JointParam::~JointParam()
{}

JointParam & JointParam::operator=(const JointParam &rhs)
{
	if (this == &rhs) return *this;
	//static_cast<base_type &>(*this) = rhs;
	jointType_ = rhs.jointType_;
	lowerLimit_ = rhs.lowerLimit_;
	upperLimit_ = rhs.upperLimit_;
	maxJointSpeed_ = rhs.maxJointSpeed_;
	//reductionRatio_ = rhs.reductionRatio_;
	//encoderResolution_ = rhs.encoderResolution_;
	return *this;
}

}  // namespace swl
