#include "swl/Config.h"
#include "swl/kinematics/StanfordArmKinematics.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------------
// class StanfordArmKinematics

StanfordArmKinematics::StanfordArmKinematics()
: base_type()
{
	//setDeviceType(ROBOT_STANFORD_ARM);
/*
	// D-H Notation of Fu's Book, pp. 38
	addDHParam(DHParam(JointParam(JointParam::JOINT_REVOLUTE,  MathUtil::toRad(-160.0), MathUtil::toRad(160.0), MathUtil::toRad(100.0)),
		0.580, -MathConstant::PI_2, 0.0, -MathConstant::PI_2));
	addDHParam(DHParam(JointParam(JointParam::JOINT_REVOLUTE,  MathUtil::toRad(-45.0),  MathUtil::toRad(225.0), MathUtil::toRad(100.0)),
		0.290, -MathConstant::PI_2, 0.0, MathConstant::PI_2));
	addDHParam(DHParam(JointParam(JointParam::JOINT_PRISMATIC, -0.200,                 0.500,                 0.2),
		0.645, -MathConstant::PI_2, 0.0, 0.0));
	addDHParam(DHParam(JointParam(JointParam::JOINT_REVOLUTE,  MathUtil::toRad(-160.0), MathUtil::toRad(160.0), MathUtil::toRad(200.0)),
		0.0,   0.0,               0.0, -MathConstant::PI_2));
	addDHParam(DHParam(JointParam(JointParam::JOINT_REVOLUTE,  MathUtil::toRad(-90.0),  MathUtil::toRad(90.0),  MathUtil::toRad(200.0)),
		0.0,   0.0,               0.0, MathConstant::PI_2));
	addDHParam(DHParam(JointParam(JointParam::JOINT_REVOLUTE,  MathUtil::toRad(-270.0), MathUtil::toRad(270.0), MathUtil::toRad(200.0)),
		0.205, 0.0,               0.0, 0.0));
*/
}

StanfordArmKinematics::~StanfordArmKinematics()
{
}

bool StanfordArmKinematics::calcJacobian(const StanfordArmKinematics::coords_type &jointCoords, Matrix<double> &jacobian)
// jointCoords: joint values in joint coordinates, [rad]
{
	return true;
}

double StanfordArmKinematics::calcJacobianDeterminant(const StanfordArmKinematics::coords_type &poseCoords, const bool isJointSpace /*= true*/)
// if isJointSpace == true,
//  poseCoords: joint values in joint coordinates, [rad]
// if isJointSpace == false,
//   poseCoords: cartesian values in cartesian coordinates, [m ; rad]
{
	return 0.0;
}

TMatrix3<double> StanfordArmKinematics::doCalcDHMatrix(const double d_i, const double theta_i, const double a_i, const double alpha_i)
// D-H Notation of Asada's Book, A(i-1,i)
// 현재 index of axes가 0부터 시작되므로 base frame of a robot의 index는 -1이다
{
	return base_type::doCalcDHMatrix(d_i, theta_i, a_i, alpha_i);
}

TMatrix3<double> StanfordArmKinematics::doJointCoordsToTMatrix(const StanfordArmKinematics::coords_type& jointCoords)
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
	const DHParam &dh1 = getDHParam(1);
	dhMat *= doCalcDHMatrix(dh1.getD(), jointCoords[1] + dh1.getTheta(), dh1.getA(), dh1.getAlpha());
	const DHParam &dh2 = getDHParam(2);
	dhMat *= doCalcDHMatrix(jointCoords[2] + dh2.getD(), dh2.getTheta(), dh2.getA(), dh2.getAlpha());
	const DHParam &dh3 = getDHParam(3);
	dhMat *= doCalcDHMatrix(dh3.getD(), jointCoords[3] + dh3.getTheta(), dh3.getA(), dh3.getAlpha());
	const DHParam &dh4 = getDHParam(4);
	dhMat *= doCalcDHMatrix(dh4.getD(), jointCoords[4] + dh4.getTheta(), dh4.getA(), dh4.getAlpha());
	const DHParam &dh5 = getDHParam(5);
	dhMat *= doCalcDHMatrix(dh5.getD(), jointCoords[5] + dh5.getTheta(), dh5.getA(), dh5.getAlpha());

	return dhMat;
}

TMatrix3<double> StanfordArmKinematics::doCartesianCoordsToTMatrix(const StanfordArmKinematics::coords_type &cartesianCoords)
// be used in inverse kinematics
{
	return base_type::doCartesianCoordsToTMatrix(cartesianCoords);
}

bool StanfordArmKinematics::doTMatrixToCartesianCoords(const TMatrix3<double> &tmatrix, StanfordArmKinematics::coords_type &cartesianCoords, const StanfordArmKinematics::coords_type *refCartesianCoordsPtr /*= NULL*/)
// be used in forward kinematics
// fixed angle, X(alpha) => Y(beta) => Z(gamma): R = Rz(gamma) * Ry(beta) * Rx(alpba) from Craig's Book
// tmatrix: the 4x4 transformation matrix of position & orientation of the end-effector of a robot
// cartesianCoords: result of forward kinematics expressed by a cartesian coordinates, [m ; rad]
// refCartesianCoordsPtr: in solving forward kinematics, reference value to decide the resultant cartesian values of forward kinematics
//  if a robot is of serial type, refCartesianCoordsPtr == NULL (no need)
//  if a robot is of in-parallel type, refCartesianCoordsPtr != NULL
{
	if (cartesianCoords.size() != 6) cartesianCoords.resize(6);

	// x
	cartesianCoords[0] = tmatrix.T().x();
	// y
	cartesianCoords[1] = tmatrix.T().y();
	// z
	cartesianCoords[2] = tmatrix.T().z();

	// alpha, beta & gamma
	return doCalcRotationAngle(tmatrix, cartesianCoords[3], cartesianCoords[4], cartesianCoords[5]);
}

bool StanfordArmKinematics::doTMatrixToJointCoords(const TMatrix3<double> &tmatrix, StanfordArmKinematics::coords_type &jointCoords, const StanfordArmKinematics::coords_type *refJointCoordsPtr /*= NULL*/)
// be used in inverse kinematics
// 현재 index of axes가 0부터 시작되므로 base frame of a robot의 index는 -1이다
// tmatrix: the 4x4 transformation matrix of position & orientation of the end-effector of a robot
// jointCoords: result of inverse kinematics expressed by a joint coordinates, [rad]
// refJointCoordsPtr: in solving inverse kinematics, reference value to decide the resultant joint values of inverse kinematics
//  if a robot is of serial type, refCartesianCoordsPtr != 0L
//  if a robot is of in-parallel type, refCartesianCoordsPtr == 0L ( no need )
{
	if (jointCoords.size() < getDOF()) jointCoords.resize(getDOF());

	const double nx = tmatrix.X().x(), ny = tmatrix.X().y(), nz = tmatrix.X().z();
	const double tx = tmatrix.Y().x(), ty = tmatrix.Y().y(), tz = tmatrix.Y().z();
	const double bx = tmatrix.Z().x(), by = tmatrix.Z().y(), bz = tmatrix.Z().z();
	const double px = tmatrix.T().x(), py = tmatrix.T().y(), pz = tmatrix.T().z();

	const double d1 = getDHParam(0).getD(), d2 = getDHParam(1).getD(),
		   d3 = getDHParam(2).getD(), d6 = getDHParam(5).getD();

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
#endif  // _DEBUG
*/
	return true;
}

TMatrix3<double> StanfordArmKinematics::doCalcRMatrix(const double alpha, const double beta, const double gamma)
// be used in inverse kinematics
// fixed angle, X(alpha) => Y(beta) => Z(gamma): R = Rz(gamma) * Ry(beta) * Rx(alpba) from Craig's Book
{
	return base_type::doCalcRMatrix(alpha, beta, gamma);
}

bool StanfordArmKinematics::doCalcRotationAngle(const TMatrix3<double> &tmatrix, double &alpha, double &beta, double &gamma)
// be used in forward kinematics
// fixed angle, X(alpha) => Y(beta) => Z(gamma): R = Rz(gamma) * Ry(beta) * Rx(alpba) from Craig's Book
{
	return base_type::doCalcRotationAngle(tmatrix, alpha, beta, gamma);
}

}  // namespace swl
