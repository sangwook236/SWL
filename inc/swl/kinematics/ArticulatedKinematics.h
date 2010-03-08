#if !defined(__SWL_KINEMATICS__ARTICULATED_KINEMATICS__H_)
#define __SWL_KINEMATICS__ARTICULATED_KINEMATICS__H_ 1


#include "swl/kinematics/RobotKinematics.h"


namespace swl {

//--------------------------------------------------------------------------------
// class ArticulatedKinematics

class SWL_KINEMATICS_API ArticulatedKinematics: public RobotKinematics
{
public:
	typedef RobotKinematics base_type;

public:
	ArticulatedKinematics();
	virtual ~ArticulatedKinematics();

private:
	ArticulatedKinematics(const ArticulatedKinematics &rhs);
	ArticulatedKinematics& operator=(const ArticulatedKinematics &rhs);

public:
	///
	/*final*/ /*virtual*/ bool solveForward(const coords_type &jointCoords, coords_type &cartesianCoords, const coords_type *refCartesianCoordsPtr = NULL);
	/*final*/ /*virtual*/ bool solveInverse(const coords_type &cartesianCoords, coords_type &jointCoords, const coords_type *refJointCoordsPtr = NULL);

protected:
	///
	tmatrix_type doCalcDHMatrix(const size_t axisId, const double variableJointValue);
	virtual tmatrix_type doCalcDHMatrix(const double d_i, const double theta_i, const double a_i, const double alpha_i) = 0;  // implemented

	///
	virtual tmatrix_type doJointCoordsToTMatrix(const coords_type &jointCoords) = 0;  // implemented
	virtual tmatrix_type doCartesianCoordsToTMatrix(const coords_type &cartesianCoords) = 0;  // implemented

	///
	virtual bool doTMatrixToCartesianCoords(const tmatrix_type &tmatrix, coords_type &cartesianCoords, const coords_type *refCartesianCoordsPtr = NULL) = 0;
	virtual bool doTMatrixToJointCoords(const tmatrix_type &tmatrix, coords_type &jointCoords, const coords_type *refJointCoordsPtr = NULL) = 0;

	///
	virtual tmatrix_type doCalcRMatrix(const double alpha, const double beta, const double gamma) = 0;  // implemented
	virtual bool doCalcRotationAngle(const tmatrix_type &tmatrix, double &alpha, double &beta, double &gamma) = 0;  // implemented
//@}
};

}  // namespace swl


#endif  // __SWL_KINEMATICS__ARTICULATED_KINEMATICS__H_
