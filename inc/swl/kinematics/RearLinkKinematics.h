#if !defined(__SWL_KINEMATICS__REAR_LINK_KINEMATICS__H_)
#define __SWL_KINEMATICS__REAR_LINK_KINEMATICS__H_ 1


#include "swl/kinematics/ArticulatedKinematics.h"


namespace swl {

//--------------------------------------------------------------------------------
// class RearLinkKinematics

class SWL_KINEMATICS_API RearLinkKinematics: public ArticulatedKinematics
{
public:
	typedef ArticulatedKinematics base_type;

public:
	RearLinkKinematics();
	virtual ~RearLinkKinematics();

private:
	RearLinkKinematics(const RearLinkKinematics &rhs);
	RearLinkKinematics & operator=(const RearLinkKinematics &rhs);

public:
	///
	/*virtual*/ bool isReachable(const coords_type &poseCoords, const bool isJointSpace = true);

	///
	/*virtual*/ bool calcJacobian(const coords_type &jointCoords, Matrix<double> &jacobian);
	/*virtual*/ double calcJacobianDeterminant(const coords_type &poseCoords, const bool isJointSpace = true);

protected:
	///
	/*virtual*/ TMatrix3<double> doCalcDHMatrix(const double d_i, const double theta_i, const double a_i, const double alpha_i);
	/*virtual*/ TMatrix3<double> doJointCoordsToTMatrix(const coords_type &jointCoords);
	/*virtual*/ TMatrix3<double> doCartesianCoordsToTMatrix(const coords_type &cartesianCoords);

	///
	/*virtual*/ bool doTMatrixToCartesianCoords(const TMatrix3<double> &tmatrix, coords_type &cartesianCoords, const coords_type *refCartesianCoordsPtr = NULL);
	/*virtual*/ bool doTMatrixToJointCoords(const TMatrix3<double> &tmatrix, coords_type &jointCoords, const coords_type *refJointCoordsPtr = NULL);

	///
	/*virtual*/ TMatrix3<double> doCalcRMatrix(const double alpha, const double beta, const double gamma);
	/*virtual*/ bool doCalcRotationAngle(const TMatrix3<double> &tmatrix, double &alpha, double &beta, double &gamma);
};

}  // namespace swl


#endif  // __SWL_KINEMATICS__REAR_LINK_KINEMATICS__H_
