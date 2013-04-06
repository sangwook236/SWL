#include "swl/Config.h"
#include "swl/kinematics/ScaraKinematics.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------------
// class Scara2Kinematics

Scara2Kinematics::Scara2Kinematics()
: base_type()
{
	setDeviceType(DEVICE_ROBOT_SCARA);
}

Scara2Kinematics::~Scara2Kinematics()
{
}

bool Scara2Kinematics::solveForward(const Scara2Kinematics::coords_type &jointCoords, Scara2Kinematics::coords_type &cartesianCoords, const Scara2Kinematics::coords_type *refCartesianCoordsPtr /*= NULL*/)
// jointCoords: input values expressed by a joint coordinates, [rad]
// cartesianCoords: result of forward kinematics expressed by a cartesian coordinates, [m ; rad]
// refCartesianCoordsPtr: in solving forward kinematics, reference value to decide the resultant cartesian values of forward kinematics
//  if a robot is of serial type, refCartesianCoordsPtr == NULL (no need)
//  if a robot is of in-parallel type, refCartesianCoordsPtr != NULL
{
	const size_t nDOF = getDOF();
	if (jointCoords.size() < 2 || nDOF < 2) return false;
	if (cartesianCoords.size() < nDOF) cartesianCoords.resize(nDOF);

	// properties
	const double a1 = getDHParam(0).getA();
	const double a2 = getDHParam(1).getA();

	cartesianCoords[0] = a1 * std::cos(jointCoords[0]) + a2 * std::cos(jointCoords[1]);  // x
	cartesianCoords[1] = a1 * std::sin(jointCoords[0]) + a2 * std::sin(jointCoords[1]);  // y

	return true;
}

bool Scara2Kinematics::solveInverse(const Scara2Kinematics::coords_type &cartesianCoords, Scara2Kinematics::coords_type &jointCoords, const Scara2Kinematics::coords_type *refJointCoordsPtr /*= NULL*/)
// cartesianCoords: input values expressed by a cartesian coordinates, [rad ; m]
// jointCoords: result of inverse kinematics expressed by a joint coordinates, [rad]
// refJointCoordsPtr: in solving inverse kinematics, reference value to decide the resultant joint values of inverse kinematics
//  if a robot is of serial type, refCartesianCoordsPtr != NULL
//  if a robot is of in-parallel type, refCartesianCoordsPtr == NULL (no need)
{
	const size_t nDOF = getDOF();
	if (cartesianCoords.size() < 2 || nDOF < 2) return false;
	if (jointCoords.size() < nDOF) jointCoords.resize(nDOF);

	// properties
	const double a1 = getDHParam(0).getA();
	const double a2 = getDHParam(1).getA();

	const double px = cartesianCoords[0];  // x
	const double py = cartesianCoords[1];  // y

	//
	double psi1, psi2;
	const double dist = px*px + py*py;
	const double alpha = std::atan2(py, px);

	if (dist > (a1+a2)*(a1+a2) || dist < (a1-a2)*(a1-a2)) return false;

	// calculate theta2
	double L = (dist - a1*a1 - a2*a2) / (2.0 * a1 * a2);
	if (L > 1.0) L = 1.0;
	else if (L < -1.0) L = -1.0;

	// because of the positive rotational direction of a 2nd axis
	const double tmp = 2.0 * std::atan2(std::sqrt(1 - L), std::sqrt(1 + L));
	psi1 = tmp;
	psi2 = -tmp;
	if (refJointCoordsPtr)
	{
		if (fabs(psi1 - (*refJointCoordsPtr)[1]) <= fabs(psi2 - (*refJointCoordsPtr)[1]))
			jointCoords[1] = psi1;
		else
			jointCoords[1] = psi2;
	}
	else
	{
		jointCoords[1] = psi1;
		//jointCoords[1] = psi2;
	}

	if (!checkJointLimit(1, jointCoords[1])) return false;

	const bool bIsUpperArm = jointCoords[1] >= 0 ? true : false;

	// calculate theta1
	L = a2 * std::sin(jointCoords[1]) / std::sqrt(dist);

	if (bIsUpperArm)
	{
		if (L < -MathConstant::EPS || L > 1) return false;
		else if (fabs(L) <= MathConstant::EPS) jointCoords[0] = alpha;
		else
		{
			double u = -2.0 / L;  // 0 < L <= 1  ==>  u <= -2
			double det = u * u - 4.0;  // u <= -2  ==>  det >= 0

			// solution of a quadratic equation when u < 0
			//psi1 = 2.0 * std::atan((-u + std::sqrt(det)) / 2.0);
			//psi2 = 2.0 * std::atan(-2.0 / (u - std::sqrt(det)));
			psi1 = std::fmod(2.0 * std::atan((-u + std::sqrt(det)) / 2.0), MathConstant::_2_PI);
			psi2 = std::fmod(2.0 * std::atan(-2.0 / (u - std::sqrt(det))), MathConstant::_2_PI);
		}
	}
	else
	{
		if (L < -1 || L > MathConstant::EPS) return false;
		else if (fabs(L) <= MathConstant::EPS) jointCoords[0] = alpha;
		else
		{
			const double u = -2.0 / L;  // -1 <= L < 0  ==>  u >= 2
			const double det = u * u - 4.0;  // u >= 2  ==>  det >= 0

			// solution of a quadratic equation when u > 0
			//psi1 = 2.0 * std::atan(-2.0 / (u + std::sqrt(det)));
			//psi2 = 2.0 * std::atan((-u - std::sqrt(det)) / 2.0);
			psi1 = std::fmod(2.0 * std::atan(-2.0 / (u + std::sqrt(det))), MathConstant::_2_PI);
			psi2 = std::fmod(2.0 * std::atan((-u - std::sqrt(det)) / 2.0), MathConstant::_2_PI);
		}
	}

	if ((psi1 < 0 || psi1 > MathConstant::PI) && (psi2 < 0 || psi2 > MathConstant::PI))
		return false;
	else if (psi1 < 0 || psi1 > MathConstant::PI)
	{
		// because of the positive rotational direction of a 1st axis
		//jointCoords[0] = alpha - psi2;
		//jointCoords[0] = -(alpha - psi2);
		psi2 = std::fmod(-alpha - psi2, MathConstant::_2_PI);
		if (psi2 > MathConstant::PI) psi2 = psi2 - MathConstant::_2_PI;
		else if (psi2 < -MathConstant::PI) psi2 = psi2 + MathConstant::_2_PI;
		jointCoords[0] = psi2;
	}
	else if (psi2 < 0 || psi2 > MathConstant::PI)
	{
		// because of the positive rotational direction of a 1st axis
		//jointCoords[0] = alpha - psi1;
		//jointCoords[0] = -(alpha - psi1);
		psi1 = std::fmod(-alpha - psi1, MathConstant::_2_PI);
		if (psi1 > MathConstant::PI) psi1 = psi1 - MathConstant::_2_PI;
		else if (psi1 < -MathConstant::PI) psi1 = psi1 + MathConstant::_2_PI;
		jointCoords[0] = psi1;
	}
	else
	{
		// because of the positive rotational direction of a 1st axis
		//psi1 = alpha - psi1;
		//psi2 = alpha - psi2;
		//psi1 = -(alpha - psi1);
		//psi2 = -(alpha - psi2);
		psi1 = std::fmod(-alpha - psi1, MathConstant::_2_PI);
		psi2 = std::fmod(-alpha - psi2, MathConstant::_2_PI);
		if (psi1 > MathConstant::PI) psi1 = psi1 - MathConstant::_2_PI;
		else if (psi1 < -MathConstant::PI) psi1 = psi1 + MathConstant::_2_PI;
		if (psi2 > MathConstant::PI) psi2 = psi2 - MathConstant::_2_PI;
		else if (psi2 < -MathConstant::PI) psi2 = psi2 + MathConstant::_2_PI;

		if (refJointCoordsPtr)
		{
			jointCoords[0] = fabs(psi1 - (*refJointCoordsPtr)[0]) <= fabs(psi2 - (*refJointCoordsPtr)[0]) ? psi1 : psi2;
		}
		else
		{
			jointCoords[0] = psi1;
			//jointCoords[0] = psi2;
		}
	}

	return checkJointLimit(0, jointCoords[0]);
}


//--------------------------------------------------------------------------------
// class Scara3Kinematics

Scara3Kinematics::Scara3Kinematics()
: base_type()
{
	setDeviceType(DEVICE_ROBOT_SCARA);
}

Scara3Kinematics::~Scara3Kinematics()
{
}

bool Scara3Kinematics::solveForward(const Scara3Kinematics::coords_type &jointCoords, Scara3Kinematics::coords_type &cartesianCoords, const Scara3Kinematics::coords_type *refCartesianCoordsPtr /*= NULL*/)
// jointCoords: input values expressed by a joint coordinates, [rad]
// cartesianCoords: result of forward kinematics expressed by a cartesian coordinates, [m ; rad]
// refCartesianCoordsPtr: in solving forward kinematics, reference value to decide the resultant cartesian values of forward kinematics
//  if a robot is of serial type, refCartesianCoordsPtr == NULL (no need)
//  if a robot is of in-parallel type, refCartesianCoordsPtr != NULL
{
	const size_t nDOF = getDOF();
	if (jointCoords.size() < 3 || nDOF < 3) return false;
	if (cartesianCoords.size() < nDOF) cartesianCoords.resize(nDOF);

	if (base_type::solveForward(jointCoords, cartesianCoords, refCartesianCoordsPtr))
	{
		cartesianCoords[2] = jointCoords[0] + jointCoords[1] + jointCoords[2];  // phi
		return true;
	}
	else return false;
}

bool Scara3Kinematics::solveInverse(const Scara3Kinematics::coords_type &cartesianCoords, Scara3Kinematics::coords_type &jointCoords, const Scara3Kinematics::coords_type *refJointCoordsPtr /*= NULL*/)
// cartesianCoords: input values expressed by a cartesian coordinates, [rad ; m]
// jointCoords: result of inverse kinematics expressed by a joint coordinates, [rad]
// refJointCoordsPtr: in solving inverse kinematics, reference value to decide the resultant joint values of inverse kinematics
//  if a robot is of serial type, refCartesianCoordsPtr != NULL
//  if a robot is of in-parallel type, refCartesianCoordsPtr == NULL (no need)
{
	const size_t nDOF = getDOF();
	if (cartesianCoords.size() < 3 || nDOF < 3) return false;
	if (jointCoords.size() < nDOF) jointCoords.resize(nDOF);

	if (base_type::solveInverse(cartesianCoords, jointCoords, refJointCoordsPtr))
	{
		// properties
		const double phi = cartesianCoords[2];  // phi

		// calculate theta3
		double psi = std::fmod(phi - jointCoords[0] - jointCoords[1], MathConstant::_2_PI);
		if (psi > MathConstant::PI) psi = psi - MathConstant::_2_PI;
		else if (psi < -MathConstant::PI) psi = psi + MathConstant::_2_PI;
		jointCoords[2] = psi;

		return checkJointLimit(2, jointCoords[2]);
	}
	else return false;
}

}  // namespace swl
