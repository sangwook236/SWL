#if !defined(__SWL_KINEMATICS__KINEMATICS__H_)
#define __SWL_KINEMATICS__KINEMATICS__H_ 1


#include "swl/kinematics/JointParam.h"
#include "swl/kinematics/ScrewAxis.h"
#include <vector>


namespace swl {

//--------------------------------------------------------------------------------
// class KinematicsBase

class SWL_KINEMATICS_API KinematicsBase
{
public:
	//typedef KinematicsBase		base_type;
	typedef std::vector<double>		coords_type;

public:
	enum ELinkage {
		LINKAGE_UNKNOWN_ = 0,  // for error
		LINKAGE_UNDEFINED,
		LINKAGE_OPEN_LOOP, LINKAGE_CLOSED_LOOP,
		LINKAGE_SERIAL, LINKAGE_IN_PARALLEL
	};

	enum EDevice {
		DEVICE_UNKNOWN = 0,  // for error
		DEVICE_UNDEFINED,
		//  robot type
		DEVICE_SERIAL_ROBOT, DEVICE_IN_PARALLEL_ROBOT,
		DEVICE_ROBOT_ARTICULATED, DEVICE_ROBOT_SCARA, DEVICE_ROBOT_CARTESIAN, DEVICE_ROBOT_GANTRY,
/*
		DEVICE_ROBOT_PUMA560, DEVICE_ROBOT_STANFORD_ARM,
		DEVICE_DAEWOO_DR06, DEVICE_DAEWOO_DR120,
		DEVICE_SAMSUNG_FARA_RAS2, DEVICE_SAMSUNG_FARA_RAT1, DEVICE_SAMSUNG_FARA_RAT2,
		DEVICE_SAMSUNG_FARA_RSS5, DEVICE_SAMSUNG_FARA_RSM5,
		DEVICE_SAMSUNG_FARA_RCM4M, DEVICE_SAMSUNG_FARA_RCM4X,
*/
		// motion controller type
		DEVICE_MOTION_CONTROLLER,
/*
		DEVICE_SAMSUNG_MMC_BOARD, 
*/
		// lathe type
		DEVICE_LATHE, DEVICE_LATHE_1AXIS, DEVICE_LATHE_2AXIS, DEVICE_LATHE_3AXIS, DEVICE_LATHE_4AXIS, DEVICE_LATHE_5AXIS,
		// milling type
		DEVICE_MILLING, DEVICE_MILLING_HORIZONTAL, DEVICE_MILLING_VERTICAL,
		// hydraulic, pneumatic & lubrication type
		DEVICE_HYDRAULIC, DEVICE_PNEUMATIC, DEVICE_LUBRICATION
	};

public:
	KinematicsBase();
	virtual ~KinematicsBase();

private:
	KinematicsBase(const KinematicsBase &rhs);
	KinematicsBase & operator=(const KinematicsBase &rhs);

public:
	///
	virtual size_t getDOF() const = 0;
	virtual JointParam & getJointParam(const size_t jointId) const = 0;

	///
	void setDeviceType(const EDevice deviceType)  {  deviceType_ = deviceType;  }
	EDevice getDeviceType() const  {  return deviceType_;  }

	///
	bool checkJointLimit(const size_t jointId, const double jointValue);
	static bool checkLimit(const double jointValue, const double lowerLimit, const double upperLimit)
	{  return lowerLimit <= jointValue && jointValue <= upperLimit;  }

	///
	virtual bool solveForward(const coords_type &aJointPose, coords_type &aCartesianPose, const coords_type *pRefCartesianPose = NULL) = 0;
	virtual bool solveInverse(const coords_type &aCartesianPose, coords_type &aJointPose, const coords_type *pRefJointPose = NULL) = 0;
	virtual coords_type cartesianToSpatial(const coords_type &aCartesianPose) = 0;
	virtual coords_type spatialToCartesian(const coords_type &aSpatialCoord) = 0;
	virtual void setRotOrder(const unsigned int order) = 0;
	virtual unsigned int getRotOrder() = 0;

protected:
	/// device type
	EDevice deviceType_;
};


//--------------------------------------------------------------------------------
// class Kinematics

class SWL_KINEMATICS_API Kinematics: public KinematicsBase
{
public:
	typedef KinematicsBase				base_type;
	typedef std::vector<ScrewAxis>		screw_ctr;

public:
	Kinematics();
	virtual ~Kinematics();

private:
	Kinematics(const Kinematics &rhs);
	Kinematics & operator=(const Kinematics &rhs);

public:
	///
	/*virtual*/ size_t getDOF() const  {  return screwAxisCtr_.size();  }
	/*virtual*/ JointParam & getJointParam(const size_t jointId) const;

	///
	void addScrewAxis(const ScrewAxis &screwAxis);
	ScrewAxis & getScrewAxis(const size_t jointId);
	const ScrewAxis & getScrewAxis(const size_t jointId) const;
	void removeScrewAxis(const size_t jointId);
	void clearScrewAxis()  {  screwAxisCtr_.clear();  }

protected:
	/// screw axis
	screw_ctr screwAxisCtr_;
};

}  // namespace swl


#endif  // __SWL_KINEMATICS__KINEMATICS__H_
