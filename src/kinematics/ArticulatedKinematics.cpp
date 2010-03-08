#include "swl/Config.h"
#include "swl/kinematics/ArticulatedKinematics.h"
#include "swl/base/LogException.h"
#include "swl/math/MathConstant.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {
	
//--------------------------------------------------------------------------------
// class ArticulatedKinematics

ArticulatedKinematics::ArticulatedKinematics()
: base_type()
{
	setDeviceType(DEVICE_ROBOT_ARTICULATED);
}

ArticulatedKinematics::~ArticulatedKinematics()
{
}

bool ArticulatedKinematics::solveForward(const ArticulatedKinematics::coords_type &jointCoords, ArticulatedKinematics::coords_type &cartesianCoords, const ArticulatedKinematics::coords_type *refCartesianCoordsPtr /*= NULL*/)
// jointCoords: input values expressed by a joint coordinates, [rad]
// cartesianCoords: result of forward kinematics expressed by a cartesian coordinates, [m ; rad]
// refCartesianCoordsPtr: in solving forward kinematics, reference value to decide the resultant cartesian values of forward kinematics
//  if a robot is of serial type, refCartesianCoordsPtr == NULL (no need)
//  if a robot is of in-parallel type, refCartesianCoordsPtr != NULL
{
	cartesianCoords.resize(jointCoords.size());
	return doTMatrixToCartesianCoords(doJointCoordsToTMatrix(jointCoords), cartesianCoords, refCartesianCoordsPtr);
}

bool ArticulatedKinematics::solveInverse(const ArticulatedKinematics::coords_type &cartesianCoords, ArticulatedKinematics::coords_type &jointCoords, const ArticulatedKinematics::coords_type *refJointCoordsPtr /*= NULL*/)
// cartesianCoords: input values expressed by a cartesian coordinates, [rad ; m]
// jointCoords: result of inverse kinematics expressed by a joint coordinates, [rad]
// refJointCoordsPtr: in solving inverse kinematics, reference value to decide the resultant joint values of inverse kinematics
//  if a robot is of serial type, refCartesianCoordsPtr != NULL
//  if a robot is of in-parallel type, refCartesianCoordsPtr == NULL (no need)
{
	jointCoords.resize(cartesianCoords.size());
	return doTMatrixToJointCoords(doCartesianCoordsToTMatrix(cartesianCoords), jointCoords, refJointCoordsPtr);
}

ArticulatedKinematics::tmatrix_type ArticulatedKinematics::doCalcDHMatrix(const size_t axisId, const double variableJointValue)
// D-H Notation of Asada's Book, A(axisId-1,axisId)
// 현재 index of axex가 0부터 시작되므로 base frame of a robot의 index는 -1이다
{
	const DHParam &dh = getDHParam(axisId);

	double d_i = 0.0, theta_i = 0.0;
	switch (getJointParam(axisId).getJointType())
	{
	case JointParam::JOINT_REVOLUTE :
		d_i = dh.getD();
		//theta_i = /*dh.getTheta() + */ variableJointValue + dh.getInitial();
		theta_i = /*dh.getTheta() + */ variableJointValue + dh.getTheta();
		break;
	case JointParam::JOINT_PRISMATIC :
		//d_i = /*dh.getD() + */ variableJointValue + dh.getInitial();
		d_i = /*dh.getD() + */ variableJointValue + dh.getD();
		theta_i = dh.getTheta();
		break;
	case JointParam::JOINT_FIXED :
	case JointParam::JOINT_HELICAL :
	case JointParam::JOINT_UNIVERSAL :
	case JointParam::JOINT_CYLINDRICAL :
	case JointParam::JOINT_PLANAR :
	case JointParam::JOINT_SPHERICAL :
	case JointParam::JOINT_UNKNOWN :
	case JointParam::JOINT_UNDEFINED :
	default :
		d_i = dh.getD();
		theta_i = dh.getTheta();
		break;
	}

	return doCalcDHMatrix(d_i, theta_i, dh.getA(), dh.getAlpha());
}

ArticulatedKinematics::tmatrix_type ArticulatedKinematics::doCalcDHMatrix(const double d_i, const double theta_i, const double a_i, const double alpha_i)
// D-H Notation of Asada's Book, A(i-1,i)
// 현재 index of axes가 0부터 시작되므로 base frame of a robot의 index는 -1이다
{
	return DHParam::toDHMatrix(DHParam(d_i, theta_i, a_i, alpha_i), false);
}

ArticulatedKinematics::tmatrix_type ArticulatedKinematics::doJointCoordsToTMatrix(const ArticulatedKinematics::coords_type &jointCoords)
// be used in forward kinematics
{
	const size_t nDOF = getDOF();
	if (nDOF < 1 || nDOF != jointCoords.size()) return tmatrix_type();

	tmatrix_type dhMat(doCalcDHMatrix(0, jointCoords[0]));
	for (size_t i = 1; i < nDOF; ++i)
		dhMat *= doCalcDHMatrix(i, jointCoords[i]);

	return dhMat;
}

ArticulatedKinematics::tmatrix_type ArticulatedKinematics::doCartesianCoordsToTMatrix(const ArticulatedKinematics::coords_type &cartesianCoords)
// be used in inverse kinematics
{
	if (cartesianCoords.size() != 6) return tmatrix_type();

	tmatrix_type tmat(doCalcRMatrix(cartesianCoords[3], cartesianCoords[4], cartesianCoords[5]));
	tmat.T().x() = cartesianCoords[0];
	tmat.T().y() = cartesianCoords[1];
	tmat.T().z() = cartesianCoords[2];

	return tmat;
}

ArticulatedKinematics::tmatrix_type ArticulatedKinematics::doCalcRMatrix(const double alpha, const double beta, const double gamma)
// be used in inverse kinematics
// fixed angle, X(alpha) => Y(beta) => Z(gamma): R = Rz(gamma) * Ry(beta) * Rx(alpba) from Craig's Book
{
	tmatrix_type rotMat;

	const double sa = std::sin(alpha), ca = std::cos(alpha);
	const double sb = std::sin(beta), cb = std::cos(beta);
	const double sr = std::sin(gamma), cr = std::cos(gamma);

	//  x
	rotMat.X().x() = cr * cb;
	rotMat.X().y() = sr * cb;
	rotMat.X().z() = -sb;

	//  y
	rotMat.Y().x() = cr * sb * sa - sr * ca;
	rotMat.Y().y() = sr * sb * sa + cr * ca;
	rotMat.Y().z() = cb * sa;

	//  z
	rotMat.Z().x() = cr * sb * ca + sr * sa;
	rotMat.Z().y() = sr * sb * ca - cr * sa;
	rotMat.Z().z() = cb * ca;

	//  p
	//rotMat.T().x() = 0.0;
	//rotMat.T().y() = 0.0;
	//rotMat.T().z() = 0.0;

	return rotMat;
}

bool ArticulatedKinematics::doCalcRotationAngle(const ArticulatedKinematics::tmatrix_type& tmatrix, double &alpha, double &beta, double &gamma)
// be used in forward kinematics
// fixed angle, X(alpha) => Y(beta) => Z(gamma): R = Rz(gamma) * Ry(beta) * Rx(alpba) from Craig's Book
{
	const double tolerance = MathConstant::EPS;

	if (std::fabs(tmatrix.X().x()) > tolerance || std::fabs(tmatrix.X().y()) > tolerance)
	{
		beta = std::atan2(-tmatrix.X().z(), std::sqrt(std::pow(tmatrix.X().x(), 2.0) + std::pow(tmatrix.X().y(), 2.0)));
		const double cb = cos(beta);
		gamma = std::atan2(tmatrix.X().y()/cb, tmatrix.X().x()/cb);
		alpha = std::atan2(tmatrix.Y().z()/cb, tmatrix.Z().z()/cb);
		//gamma = std::atan2(tmatrix.X().y(), tmatrix.X().x());
		//alpha = std::atan2(tmatrix.Y().z(), tmatrix.Z().z());
	}
	else {
#if 1
		// case 1
		beta = MathConstant::PI_2;
		gamma = 0.0;
		alpha = std::atan2(tmatrix.Y().x(), tmatrix.Y().y());
#else
		// case 2
		beta = -MathConstant::PI_2;
		gamma = 0.0;
		alpha = -std::atan2(tmatrix.Y().x(), tmatrix.Y().y());
#endif
	}

	return true;
}

}  // namespace swl
