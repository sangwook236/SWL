#if !defined(__SWL_KINEMATICS__ROBOT_KINEMATICS__H_)
#define __SWL_KINEMATICS__ROBOT_KINEMATICS__H_ 1


#include "swl/kinematics/Kinematics.h"
#include "swl/kinematics/DHParam.h"
#include "swl/math/TMatrix.h"
#include "swl/math/Matrix.h"


namespace swl {

//class TrajectoryPlanning;

//--------------------------------------------------------------------------------
// class RobotKinematics

class SWL_KINEMATICS_API RobotKinematics: public KinematicsBase
{
public:
	typedef KinematicsBase					base_type;
	typedef std::vector<DHParam>			dh_ctr;
	typedef TMatrix3<double>				tmatrix_type;
	typedef KinematicsBase::coords_type		coords_type;

public:
	RobotKinematics();
	virtual ~RobotKinematics();

private:
	RobotKinematics(const RobotKinematics &rhs);
	RobotKinematics & operator=(const RobotKinematics &rhs);

public:
	///
	/*virtual*/ size_t getDOF() const  {  return dhParamCtr_.size();  }
	/*virtual*/ JointParam & getJointParam(const size_t jointId) const;

	///
	void addDHParam(const DHParam &dhParam);
	DHParam & getDHParam(const size_t jointId);
	const DHParam & getDHParam(const size_t jointId) const;
	void removeDHParam(const size_t jointId);
	void clearDHParam()  {  dhParamCtr_.clear();  }

	///
	bool isSingular(const coords_type &poseCoords, const bool isJointSpace = true);
	virtual bool isReachable(const coords_type &poseCoords, const bool isJointSpace = true);

	virtual bool calcJacobian(const coords_type &jointCoords, Matrix<double> &jacobian) = 0;
	virtual double calcJacobianDeterminant(const coords_type &poseCoords, const bool isJointSpace = true) = 0;

protected:
	dh_ctr dhParamCtr_;
};

}  // namespace swl


#endif  // __SWL_KINEMATICS__ROBOT_KINEMATICS__H_
