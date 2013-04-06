#if !defined(__SWL_KINEMATICS__JOINT_PARAMETER__H_)
#define __SWL_KINEMATICS__JOINT_PARAMETER__H_ 1


#include "swl/kinematics/ExportKinematics.h"


namespace swl {

//-----------------------------------------------------------------------------------------
// class JointParam

class SWL_KINEMATICS_API JointParam
{
public:
	//typedef JointParam base_type;

public:
	enum EJoint {
		JOINT_UNKNOWN = 0,  // for error
		JOINT_UNDEFINED,
		// 0-dof
		JOINT_FIXED,
		// 1-dof
		JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_HELICAL, JOINT_SCREW = JOINT_HELICAL,
		// 2-dof
		JOINT_UNIVERSAL, JOINT_HOOKE = JOINT_UNIVERSAL,
		JOINT_CYLINDRICAL,
		// 3-dof
		JOINT_PLANAR, JOINT_SPHERICAL
	};

public:
	JointParam();
	//JointParam(const EJoint jointType, const double lowerLimit, const double upperLimit, const double maxJointSpeed, const double reductionRatio, const unsinged int encoderResolution);
	JointParam(const EJoint jointType, const double lowerLimit, const double upperLimit, const double maxJointSpeed);
	JointParam(const JointParam &rhs);
	virtual ~JointParam();

	JointParam & operator=(const JointParam &rhs);

public:
	/// accessor & mutator
	void setJointType(const EJoint jointType)  {  jointType_ = jointType;  }
	EJoint getJointType() const  {  return jointType_;  }

	void setLowerLimit(const double lowerLimit)  {  lowerLimit_ = lowerLimit;  }
	double getLowerLimit() const  {  return lowerLimit_;  }

	void setUpperLimit(const double upperLimit)  {  upperLimit_ = upperLimit;  }
	double getUpperLimit() const  {  return upperLimit_;  }

	void setMaxJointSpeed(const double maxJointSpeed)  {  maxJointSpeed_ = maxJointSpeed;  }
	double getMaxJointSpeed() const  {  return maxJointSpeed_;  }

	//void setReductionRatio(const double reductionRatio)  {  reductionRatio_ = reductionRatio;  }
	//double getReductionRatio() const  {  return reductionRatio_;  }

	//void setEncoderResolution(const unsigned int encoderResolution)  {  encoderResolution_ = encoderResolution;  }
	//unsigned int getEncoderResolution() const  {  return encoderResolution_;  }

protected:
	/// joint type
	EJoint jointType_;
	/// the upper & lower limits of each axis, [rad] or [m]
	double lowerLimit_, upperLimit_;
	/// the maximum joint speed of each axis, [rad/sec] or [m/sec]
	double maxJointSpeed_;
	/// the reduction(gear) ratio of each axis
	//double reductionRatio_;
	/// the encoder resolution of each axis, [pulse/rev]
	//unsigned int encoderResolution_;
};

}  // namespace swl


#endif  // __SWL_KINEMATICS__JOINT_PARAMETER__H_
