#include "swl/Config.h"
#include "swl/kinematics/PumaKinematics.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------------
// class PumaKinematics

PumaKinematics::PumaKinematics()
: base_type()
{
	//setDeviceType(ROBOT_PUMA560);
/*
	// PUMA 560: D-H Notation of Fu's Book, pp. 37
	addDHParam(DHParam(JointParam(JointParam::JOINT_REVOLUTE, MathUtil::toRad(-160.0), MathUtil::toRad(160.0), MathUtil::toRad(100.0)),
		0.0,     MathConstant::PI_2, 0.0,      -MathConstant::PI_2));
	addDHParam(DHParam(JointParam(JointParam::JOINT_REVOLUTE, MathUtil::toRad(-225.0), MathUtil::toRad(45.0),  MathUtil::toRad(100.0)),
		0.14909, 0.0,              0.4318,   0.0));
	addDHParam(DHParam(JointParam(JointParam::JOINT_REVOLUTE, MathUtil::toRad(-45.0),  MathUtil::toRad(225.0), MathUtil::toRad(100.0)),
		0.0,     MathConstant::PI_2, -0.02032, MathConstant::PI_2));
	addDHParam(DHParam(JointParam(JointParam::JOINT_REVOLUTE, MathUtil::toRad(-110.0), MathUtil::toRad(170.0), MathUtil::toRad(200.0)),
		0.43307, 0.0,              0.0,      -MathConstant::PI_2));
	addDHParam(DHParam(JointParam(JointParam::JOINT_REVOLUTE, MathUtil::toRad(-100.0), MathUtil::toRad(100.0), MathUtil::toRad(200.0)),
		0.0,     0.0,              0.0,      MathConstant::PI_2));
	addDHParam(DHParam(JointParam(JointParam::JOINT_REVOLUTE, MathUtil::toRad(-266.0), MathUtil::toRad(266.0), MathUtil::toRad(200.0)),
		0.05625, 0.0,              0.0,      0.0));
*/
}

PumaKinematics::~PumaKinematics()
{
}

bool PumaKinematics::calcJacobian(const PumaKinematics::coords_type &jointCoords, Matrix<double> &jacobian)
// jointCoords: joint values in joint coordinates, [rad]
{
	return true;
}

double PumaKinematics::calcJacobianDeterminant(const PumaKinematics::coords_type &poseCoords, const bool isJointSpace /*= true*/)
// if isJointSpace == true,
//  poseCoords: joint values in joint coordinates, [rad]
// if isJointSpace == false,
//  poseCoords: cartesian values in cartesian coordinates, [m ; rad]
{
	return 0.0;
}

TMatrix3<double> PumaKinematics::doCalcDHMatrix(const double d_i, const double theta_i, const double a_i, const double alpha_i)
// D-H Notation of Asada's Book, A(i-1,i)
// 현재 index of axes가 0부터 시작되므로 base frame of a robot의 index는 -1이다
{
	return base_type::doCalcDHMatrix(d_i, theta_i, a_i, alpha_i);
}

TMatrix3<double> PumaKinematics::doJointCoordsToTMatrix(const PumaKinematics::coords_type &jointCoords)
// be used in forward kinematics
{
/*
	// error: jointCoords의 모든 값이 제대로 넘어가지 않음. 모두 zero로 넘어감
	return base_type::doJointCoordsToTMatrix(jointCoords);
*/
	const size_t nDOF = getDOF();
	if (nDOF < 1 || nDOF != jointCoords.size()) return TMatrix3<double>();

	// FIXME [check] >> initial joint values are correctly used ???
	const DHParam &dh0 = getDHParam(0);
	TMatrix3<double> dhMat(doCalcDHMatrix(dh0.getD(), jointCoords[0] + dh0.getTheta(), dh0.getA(), dh0.getAlpha()));
	for (size_t i = 1; i < nDOF; ++i)
	{
		const DHParam &dh = getDHParam(i);
		dhMat *= doCalcDHMatrix(dh.getD(), jointCoords[i] + dh.getTheta(), dh.getA(), dh.getAlpha());
	}

	return dhMat;
}

TMatrix3<double> PumaKinematics::doCartesianCoordsToTMatrix(const PumaKinematics::coords_type &cartesianCoords)
// be used in inverse kinematics
{
	return base_type::doCartesianCoordsToTMatrix(cartesianCoords);
}

bool PumaKinematics::doTMatrixToCartesianCoords(const TMatrix3<double> &tmatrix, PumaKinematics::coords_type &cartesianCoords, const PumaKinematics::coords_type *refCartesianCoordsPtr /*= NULL*/)
// be used in forward kinematics
// fixed angle, X(getAlpha()) => Y(beta) => Z(gamma): R = Rz(gamma) * Ry(beta) * Rx(alpba) from Craig's Book
// tmatrix: the 4x4 transformation matrix of position & orientation of the end-effector of a robot
// cartesianCoords: result of forward kinematics expressed by a cartesian coordinates, [m ; rad]
// refCartesianCoordsPtr: in solving forward kinematics, reference value to decide the resultant cartesian values of forward kinematics
//  if a robot is of serial type, refCartesianCoordsPtr == NULL (no need)
//  if a robot is of in-parallel type, refCartesianCoordsPtr != NULL
{
	if (cartesianCoords.size() != 6) cartesianCoords.resize(6);

	//  x
	cartesianCoords[0] = tmatrix.T().x();
	//  y
	cartesianCoords[1] = tmatrix.T().y();
	//  z
	cartesianCoords[2] = tmatrix.T().z();

	//  getAlpha(), beta & gamma
	return doCalcRotationAngle(tmatrix, cartesianCoords[3], cartesianCoords[4], cartesianCoords[5]);
}

bool PumaKinematics::doTMatrixToJointCoords(const TMatrix3<double> &tmatrix, PumaKinematics::coords_type &jointCoords, const PumaKinematics::coords_type *refJointCoordsPtr /*= NULL*/)
// be used in inverse kinematics
// 현재 index of axes가 0부터 시작되므로 base frame of a robot의 index는 -1이다
// tmatrix: the 4x4 transformation matrix of position & orientation of the end-effector of a robot
// jointCoords: result of inverse kinematics expressed by a joint coordinates, [rad]
// refJointCoordsPtr: in solving inverse kinematics, reference value to decide the resultant joint values of inverse kinematics
//  if a robot is of serial type, refCartesianCoordsPtr != NULL
//  if a robot is of in-parallel type, refCartesianCoordsPtr == NULL (no need)
{
	const size_t nDOF = getDOF();
	if (nDOF != 6) return false;
	if (jointCoords.size() < nDOF) jointCoords.resize(nDOF);

	// from Fu's Book

	const double nx = tmatrix.X().x(), ny = tmatrix.X().y(), nz = tmatrix.X().z();
	double tx = tmatrix.Y().x(), ty = tmatrix.Y().y(), tz = tmatrix.Y().z();
	const double bx = tmatrix.Z().x(), by = tmatrix.Z().y(), bz = tmatrix.Z().z();
	const double px = tmatrix.T().x(), py = tmatrix.T().y(), pz = tmatrix.T().z();

	const double d2 = getDHParam(1).getD(), d4 = getDHParam(3).getD(), d6 = getDHParam(5).getD();
	const double a2 = getDHParam(1).getA(), a3 = getDHParam(2).getA();

	//  wrist point
	const double wx = px - d6 * bx, wy = py - d6 * by, wz = pz - d6 * bz;

	double R, r = std::sqrt(wx*wx + wy*wy - d2*d2);

	// joint 1: (-pi, pi)
	jointCoords[0] = std::atan2(-/*ARM * */wy*r - wx*d2, -/*ARM * */wx*r + wy*d2);
	if (!checkJointLimit(0, jointCoords[0])) return false;

	// joint 2: (-pi, pi)
	R = std::sqrt(wx*wx + wy*wy + wz*wz - d2*d2);
	double sa = -wz / R, ca = -/*ARM * */r / R,
		   cb = (a2*a2 + R*R - (d4*d4 + a3*a3)) / (2.0 * a2 * R), sb = std::sqrt(1.0 - cb*cb);
	jointCoords[1] = atan2(sa * cb + /*ARM * ELBOW * */ca * sb, ca * cb - /*ARM * ELBOW * */sa * sb);
	if (!checkJointLimit(1, jointCoords[1])) return false;

	// joint 3: (-pi, pi)
	r = std::sqrt(d4*d4 + a3*a3);
	ca = (a2*a2 + r*r - R*R) / (2.0 * a2 * r);
	sa = /*ARM * ELBOW * */std::sqrt(1.0 - ca*ca);
	sb = d4 / r;
	cb = std::fabs(a3) / r;
	jointCoords[2] = std::atan2(sa*cb - ca*sb, ca*cb + sa*sb);
	if (!checkJointLimit(2, jointCoords[2])) return false;

	// joint 4: (-pi, pi)
	const double s1 = std::sin(jointCoords[0]), c1 = std::cos(jointCoords[0]);
	const double s23 = std::sin(jointCoords[1]+jointCoords[2]), c23 = std::cos(jointCoords[1]+jointCoords[2]);

	const double omega = 1;
	const double M = /*WRIST * */ (omega >= 0 ? 1 : -1);
	jointCoords[3] = std::atan2(M * (c1*by - s1*bx), M * (c1*c23*bx + s1*c23*by - s23*bz));
	if (!checkJointLimit(3, jointCoords[3])) return false;

	// joint 5: (-pi, pi)
	const double s4 = std::sin(jointCoords[3]), c4 = std::cos(jointCoords[3]);
	jointCoords[4] = std::atan2((c1*c23*c4 - s1*s4)*bx + (s1*c23*c4 + c1*s4)*by - c4*s23*bz, c1*s23*bx + s1*s23*by + c23*bz);
	if (!checkJointLimit(4, jointCoords[4])) return false;

	// joint 6: (-pi, pi)
	jointCoords[5] = std::atan2((-s1*c4 - c1*c23*s4)*nx + (c1*c4 - s1*c23*s4)*ny + s4*s23*nz,
		               (-s1*c4 - c1*c23*s4)*tx + (c1*c4 - s1*c23*s4)*ty + s4*s23*tz);
	if (!checkJointLimit(5, jointCoords[5])) return false;

	if (refJointCoordsPtr)
	{
	}
	else  // if refJointCoordsPtr == NULL
	{
	}
/*
#ifdef _DEBUG
	KcTRACE("Solution  :  ");
	for (coords_type::const_iterator it = jointCoords.begin() ; it != jointCoords.end() ; ++it)
		KcTRACE("%lf   ", MathUtil::toDeg(*it));
	KcTRACE("\n");
#endif  //  _DEBUG
*/
	return true;
}

TMatrix3<double> PumaKinematics::doCalcRMatrix(const double alpha, const double beta, const double gamma)
// be used in inverse kinematics
// fixed angle, X(alpha) => Y(beta) => Z(gamma): R = Rz(gamma) * Ry(beta) * Rx(alpba) from Craig's Book
{
	return base_type::doCalcRMatrix(alpha, beta, gamma);
}

bool PumaKinematics::doCalcRotationAngle(const TMatrix3<double> &tmatrix, double &alpha, double& beta, double& gamma)
// be used in forward kinematics
// fixed angle, X(alpha) => Y(beta) => Z(gamma): R = Rz(gamma) * Ry(beta) * Rx(alpba) from Craig's Book
{
	return base_type::doCalcRotationAngle(tmatrix, alpha, beta, gamma);
}

}  // namespace swl
