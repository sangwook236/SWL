#include "swl/Config.h"
#include "swl/kinematics/CartesianKinematics.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------------
// class Cartesian2Kinematics

Cartesian2Kinematics::Cartesian2Kinematics()
: base_type()
{
	setDeviceType(DEVICE_ROBOT_CARTESIAN);
}

Cartesian2Kinematics::~Cartesian2Kinematics()
{
}

bool Cartesian2Kinematics::solveForward(const Cartesian2Kinematics::coords_type &jointCoords, Cartesian2Kinematics::coords_type &cartesianCoords, const Cartesian2Kinematics::coords_type *refCartesianCoordsPtr /*= NULL*/)
// jointCoords: input values expressed by a joint coordinates, [rad]
// cartesianCoords: result of forward kinematics expressed by a cartesian coordinates, [m ; rad]
// refCartesianCoordsPtr : in solving forward kinematics, reference value to decide the resultant cartesian values of forward kinematics
//  if a robot is of serial type, refCartesianCoordsPtr == NULL (no need)
//  if a robot is of in-parallel type, refCartesianCoordsPtr != NULL
{
	const size_t nDOF = getDOF();
	if (jointCoords.size() < 2 || nDOF < 2) return false;
	if (cartesianCoords.size() < nDOF) cartesianCoords.resize(nDOF);

	cartesianCoords[0] = jointCoords[0];  // x
	cartesianCoords[1] = jointCoords[1];  // y

	return true;
}

bool Cartesian2Kinematics::solveInverse(const Cartesian2Kinematics::coords_type &cartesianCoords, Cartesian2Kinematics::coords_type &jointCoords, const Cartesian2Kinematics::coords_type *refJointCoordsPtr /*= NULL*/)
// cartesianCoords: input values expressed by a cartesian coordinates, [rad ; m]
// jointCoords: result of inverse kinematics expressed by a joint coordinates, [rad]
// refJointCoordsPtr : in solving inverse kinematics, reference value to decide the resultant joint values of inverse kinematics
//  if a robot is of serial type, refCartesianCoordsPtr != NULL
//  if a robot is of in-parallel type, refCartesianCoordsPtr == NULL (no need)
{
	const size_t nDOF = getDOF();
	if (cartesianCoords.size() < 2 || nDOF < 2) return false;
	if (jointCoords.size() < nDOF) jointCoords.resize(nDOF);

	jointCoords[0] = cartesianCoords[0];  // x
	if (!checkJointLimit(0, jointCoords[0])) return false;
	jointCoords[1] = cartesianCoords[1];  // y
	if (!checkJointLimit(1, jointCoords[1])) return false;

	return true;
}


//--------------------------------------------------------------------------------
// class Cartesian3Kinematics

Cartesian3Kinematics::Cartesian3Kinematics()
: base_type()
{
}

Cartesian3Kinematics::~Cartesian3Kinematics()
{
}

bool Cartesian3Kinematics::solveForward(const Cartesian3Kinematics::coords_type &jointCoords, Cartesian3Kinematics::coords_type &cartesianCoords, const Cartesian3Kinematics::coords_type *refCartesianCoordsPtr /* = NULL*/)
// jointCoords : input values expressed by a joint coordinates, [rad]
// cartesianCoords : result of forward kinematics expressed by a cartesian coordinates,  m ; rad]
// refCartesianCoordsPtr : in solving forward kinematics, reference value to decide the resultant cartesian values of forward kinematics
//	 if a robot is of serial type, refCartesianCoordsPtr == NULL ( no need)
//	 if a robot is of in-parallel type, refCartesianCoordsPtr != NULL
{
	const size_t nDOF = getDOF();
	if (jointCoords.size() < 3 || nDOF < 3) return false;
	if (cartesianCoords.size() < nDOF) cartesianCoords.resize(nDOF);

	const bool result = base_type::solveForward(jointCoords, cartesianCoords, refCartesianCoordsPtr);
	cartesianCoords[2] = jointCoords[2];  // z
	return result;
}

bool Cartesian3Kinematics::solveInverse(const Cartesian3Kinematics::coords_type &cartesianCoords, Cartesian3Kinematics::coords_type &jointCoords, const Cartesian3Kinematics::coords_type *refJointCoordsPtr /*= NULL*/)
// cartesianCoords : input values expressed by a cartesian coordinates, [rad ; m]
// jointCoords : result of inverse kinematics expressed by a joint coordinates, [rad]
// refJointCoordsPtr : in solving inverse kinematics, reference value to decide the resultant joint values of inverse kinematics
//  if a robot is of serial type, refCartesianCoordsPtr != NULL
//  if a robot is of in-parallel type, refCartesianCoordsPtr == NULL (no need)
{
	const size_t nDOF = getDOF();
	if (cartesianCoords.size() < 3 || nDOF < 3) return false;
	if (jointCoords.size() < nDOF) jointCoords.resize(nDOF);

	const bool result = base_type::solveInverse(cartesianCoords, jointCoords, refJointCoordsPtr);
	jointCoords[2] = cartesianCoords[2];  // z
	if (!checkJointLimit(2, jointCoords[2])) return false;
	return result;
}

}  // namespace swl
