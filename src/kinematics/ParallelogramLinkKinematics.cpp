#include "swl/Config.h"
#include "swl/kinematics/ParallelogramLinkKinematics.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------------
// class ParallelogramLinkKinematics

ParallelogramLinkKinematics::ParallelogramLinkKinematics()
: base_type()
{
/*
	// Daewoo DR06 or Samsung Faraman AS2
	// Samsung Faraman AS2's D-H Notation
	addDHParam(DHParam(JointParam(JointParam::JOINT_REVOLUTE, MathUtil::toRad(-160.0), MathUtil::toRad(160.0), MathUtil::toRad(112.5)),
		0.475, 0.0,               0.150, -MathConstant::PI_2));
	addDHParam(DHParam(JointParam(JointParam::JOINT_REVOLUTE, MathUtil::toRad(-90.0),  MathUtil::toRad(150.0), MathUtil::toRad(112.5)),
		0.0,   -MathConstant::PI_2, 0.350, 0.0));
	addDHParam(DHParam(JointParam(JointParam::JOINT_REVOLUTE, MathUtil::toRad(-150.0), MathUtil::toRad(135.0), MathUtil::toRad(112.5)),
		0.0,   0.0,               0.100, -MathConstant::PI_2));
	addDHParam(DHParam(JointParam(JointParam::JOINT_REVOLUTE, MathUtil::toRad(-140.0), MathUtil::toRad(140.0), MathUtil::toRad(225.0)),
		0.350, 0.0,               0.0,   MathConstant::PI_2));
	addDHParam(DHParam(JointParam(JointParam::JOINT_REVOLUTE, MathUtil::toRad(-120.0), MathUtil::toRad(120.0), MathUtil::toRad(200.0)),
		0.0,   MathConstant::PI,    0.0,   MathConstant::PI_2));
	addDHParam(DHParam(JointParam(JointParam::JOINT_REVOLUTE, MathUtil::toRad(-360.0), MathUtil::toRad(360.0), MathUtil::toRad(360.0)),
		0.095,                    0.0,   0.0,   0.0));
*/
}

ParallelogramLinkKinematics::~ParallelogramLinkKinematics()
{
}

bool ParallelogramLinkKinematics::isReachable(const ParallelogramLinkKinematics::coords_type &poseCoords, const bool isJointSpace /*= true*/)
// if isJointSpace == true,
//  poseCoords: joint values in joint coordinates, [rad]
// if isJointSpace == false,
//  poseCoords: cartesian values in cartesian coordinates, [m ; rad]
// if reachable, return true
{
	if (isJointSpace)  // joint coordinates
	{
		for (size_t i = 0; i < getDOF(); ++i)
			if (!checkJointLimit(i, poseCoords[i])) return false;

		// rear link의 간섭에 의한 theta2 & theta3'의 각도 제한, -59.0 deg <= theta3' - theta2 <= 59.0 deg
		if (!checkLimit(poseCoords[2]-poseCoords[1], MathUtil::toRad(-59.0), MathUtil::toRad(59.0))) return false;
	}
	else  // cartesian coordinates
	{
		coords_type jointCoords;
		return solveInverse(poseCoords, jointCoords);
	}

	return true;
}

bool ParallelogramLinkKinematics::calcJacobian(const ParallelogramLinkKinematics::coords_type &jointCoords, Matrix<double> &jacobian)
//jointCoords: joint values in joint coordinates, [rad]
{
/*
	const size_t nDOF = getDOF();
	DHParam* pDHParam;

	coords_type vCartesian(nDOF);
	if (!solveForward(jointCoords, vCartesian)) return false;
	HVector3<double> vecEE;  // position of end-effector
	vecEE.x() = vCartesian[0];  vecEE.y() = vCartesian[1];  vecEE.z() = vCartesian[2];  vecEE.w() = 1.0;

	Vector3<double> vecB, vecDir;
	HVector3<double> vecX, vecPos;
	vecB.x() = 0.0;  vecB.y() = 0.0;  vecB.z() = 1.0;
	vecX.x() = 0.0;  vecX.y() = 0.0;  vecX.z() = 0.0;  vecX.w() = 1.0;

	// direction of joint
	jacobian(3,0) = vecB.x();
	jacobian(4,0) = vecB.y();
	jacobian(5,0) = vecB.z();
	// position vector
	jacobian(0,0) = vecB.y()*vecEE.z() - vecB.z()*vecEE.y();
	jacobian(1,0) = vecB.z()*vecEE.x() - vecB.x()*vecEE.z();
	jacobian(2,0) = vecB.x()*vecEE.y() - vecB.y()*vecEE.x();

	TMatrix3<double> TM;
	for (size_t i = 1; i < nDOF; ++i)
	{
		pDHParam = getDHParam(i - 1);
		if (i - 1 == 2)
			// (DH notation 상에서의 3축의 각도) = (3축 motor의 회전각) - (2축 motor의 회전각)
			TM *= doCalcDHMatrix(pDHParam->getD(), jointCoords[i-1]-jointCoords[i-2]+pDHParam->getInitial(), pDHParam->getA(), pDHParam->getAlpha());
		else
			TM *= doCalcDHMatrix(pDHParam->getD(), jointCoords[i-1]+pDHParam->getInitial(), pDHParam->getA(), pDHParam->getAlpha());

		// direction of joint
		vecDir = TM[0] * vecB.x() + TM[1] * vecB.y() + TM[2] * vecB.z();
		jacobian(3,i) = vecDir.x();
		jacobian(4,i) = vecDir.y();
		jacobian(5,i) = vecDir.z();
		// position vector
		vecPos = vecEE - TM * vecX;
		jacobian(0,i) = vecDir.y()*vecPos.z() - vecDir.z()*vecPos.y();
		jacobian(1,i) = vecDir.z()*vecPos.x() - vecDir.x()*vecPos.z();
		jacobian(2,i) = vecDir.x()*vecPos.y() - vecDir.y()*vecPos.x();
	}
*/
	return true;
}

double ParallelogramLinkKinematics::calcJacobianDeterminant(const ParallelogramLinkKinematics::coords_type &poseCoords, const bool isJointSpace /*= true*/)
// if isJointSpace == true,
//  poseCoords: joint values in joint coordinates, [rad]
// if isJointSpace == false,
//  poseCoords: cartesian values in cartesian coordinates, [m ; rad]
{
	coords_type jointCoords;
	if (isJointSpace)  // joint coordinates
	{
		for (size_t i = 0; i < getDOF(); ++i) jointCoords.push_back(poseCoords[i]);
	}
	else  // cartesian coordinates
	{
		if (!solveInverse(poseCoords, jointCoords)) return 0.0;
	}

	const double s1 = std::sin(jointCoords[0]), c1 = std::cos(jointCoords[0]);
	const double s2 = std::sin(jointCoords[1]), c2 = std::cos(jointCoords[1]);
	const double s3 = std::sin(jointCoords[2]), c3 = std::cos(jointCoords[2]);
	const double s4 = std::sin(jointCoords[3]), c4 = std::cos(jointCoords[3]);
	const double s5 = std::sin(jointCoords[4]), c5 = std::cos(jointCoords[4]);

	double s[6][3];
	s[0][0] = 0.0;  s[0][1] = 0.0;  s[0][2] = 1.0;
	s[1][0] = -s1;  s[1][1] = c1;  s[1][2] = 0.0;
	s[2][0] = -s1;  s[2][1] = c1;  s[2][2] = 0.0;
	s[3][0] = c1*c3;  s[3][1] = s1*c3;  s[3][2] = -s3;
	s[4][0] = -s1*c4 + c1*s3*s4;  s[4][1] = c1*c4 + s1*s3*s4;  s[4][2] = c3*s4;
	s[5][0] = -s1*s4*s5 + c1*(c3*c5 - s3*c4*s5);  s[5][1] = c1*s4*s5 + s1*(c3*c5 - s3*c4*s5);  s[5][2] = -s3*c5 - c3*c4*s5;

	double rw[3];
	rw[0] = 0.050 * c1 * (3.0 + 7.0*(s2+c3) + 2.0*s3);
	rw[1] = 0.050 * s1 * (3.0 + 7.0*(s2+c3) + 2.0*s3);
	rw[2] = 0.025 * (19.0 + 14.0*(c2-s3) + 4.0*c3);

	double r[3][3];
	r[0][0] = 0.0 - rw[0];  r[0][1] = 0.0 - rw[1];  r[0][2] = 0.0 - rw[2];
	r[1][0] = 0.150*c1 - rw[0];  r[1][1] = 0.150*s1 - rw[1];  r[1][2] = 0.475 - rw[2];
	r[2][0] = 0.050*c1*(3.0+7.0*s2) - rw[0];  r[2][1] = 0.050*s1*(3.0+7.0*s2) - rw[1];  r[2][2] = 0.475+0.350*c2 - rw[2];

	return s[0][2] * (s[3][0]*( s[4][1]*s[5][2]-s[5][1]*s[4][2]) - s[4][0]*(s[3][1]*s[5][2]-s[5][1]*s[3][2]) + s[5][0]*(s[3][1]*s[4][2]-s[4][1]*s[3][2]))
		   * (s[1][1]*(s[2][1]*r[0][0]*(r[2][0]*r[1][2]-r[1][0]*r[2][2]) + s[2][0]*(r[1][0]*r[0][1]*r[2][2]-r[0][0]*r[2][1]*r[1][2]))
		   + s[1][0]*(s[2][0]*r[0][1]*(r[2][1]*r[1][2]-r[1][1]*r[2][2]) + s[2][1]*(r[0][0]*r[1][1]*r[2][2]-r[2][0]*r[0][1]*r[1][2])));
}

TMatrix3<double> ParallelogramLinkKinematics::doCalcDHMatrix(const double d_i, const double theta_i, const double a_i, const double alpha_i)
// D-H Notation of Asada's Book, A(i-1,i)
// 현재 index of axes가 0부터 시작되므로 base frame of a robot의 index는 -1이다
{
	return base_type::doCalcDHMatrix(d_i, theta_i, a_i, alpha_i);
}

TMatrix3<double> ParallelogramLinkKinematics::doJointCoordsToTMatrix(const ParallelogramLinkKinematics::coords_type &jointCoords)
// be used in forward kinematics
{
/*
	// error: jointCoords의 모든 값이 제대로 넘어가지 않음. 모두 zero로 넘어감
	TMatrix3<double> dhMat(doCalcDHMatrix(0, jointCoords[0]));
	dhMat *= doCalcDHMatrix(1, jointCoords[1]);
	// (DH notation 상에서의 3축의 각도) = (3축 motor의 회전각) - (2축 motor의 회전각)
	dhMat *= doCalcDHMatrix(2, jointCoords[2]-jointCoords[1]);
	dhMat *= doCalcDHMatrix(3, jointCoords[3]);
	dhMat *= doCalcDHMatrix(4, jointCoords[4]);
	dhMat *= doCalcDHMatrix(5, jointCoords[5]);
*/
	const size_t nDOF = getDOF();
	if (nDOF < 1 || nDOF != jointCoords.size()) return TMatrix3<double>();

	// FIXME [check] >> initial joint values are correctly used ???
	const DHParam &dh0 = getDHParam(0);
	TMatrix3<double> dhMat(doCalcDHMatrix(dh0.getD(), jointCoords[0] + dh0.getTheta(), dh0.getA(), dh0.getAlpha()));
	const DHParam &dh1 = getDHParam(1);
	dhMat *= doCalcDHMatrix(dh1.getD(), jointCoords[1] + dh1.getTheta(), dh1.getA(), dh1.getAlpha());
	// (DH notation 상에서의 3축의 각도) = (3축 motor의 회전각) - (2축 motor의 회전각)
	const DHParam &dh2 = getDHParam(2);
	dhMat *= doCalcDHMatrix(dh2.getD(), jointCoords[2] - jointCoords[1] + dh2.getTheta(), dh2.getA(), dh2.getAlpha());
	const DHParam &dh3 = getDHParam(3);
	dhMat *= doCalcDHMatrix(dh3.getD(), jointCoords[3] + dh3.getTheta(), dh3.getA(), dh3.getAlpha());
	const DHParam &dh4 = getDHParam(4);
	dhMat *= doCalcDHMatrix(dh4.getD(), jointCoords[4] + dh4.getTheta(), dh4.getA(), dh4.getAlpha());
	const DHParam &dh5 = getDHParam(5);
	dhMat *= doCalcDHMatrix(dh5.getD(), jointCoords[5] + dh5.getTheta(), dh5.getA(), dh5.getAlpha());

	return dhMat;
}

TMatrix3<double> ParallelogramLinkKinematics::doCartesianCoordsToTMatrix(const ParallelogramLinkKinematics::coords_type &cartesianCoords)
// be used in inverse kinematics
{
	return base_type::doCartesianCoordsToTMatrix(cartesianCoords);
}

bool ParallelogramLinkKinematics::doTMatrixToCartesianCoords(const TMatrix3<double> &tmatrix, ParallelogramLinkKinematics::coords_type &cartesianCoords, const ParallelogramLinkKinematics::coords_type *refCartesianCoordsPtr /*= NULL*/)
// be used in forward kinematics
// fixed angle, X(alpha) => Y(beta) => Z(gamma): R = Rz(gamma) * Ry(beta) * Rx(alpba) from Craig's Book
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

	//  alpha, beta & gamma
	return doCalcRotationAngle(tmatrix, cartesianCoords[3], cartesianCoords[4], cartesianCoords[5]);
}

bool ParallelogramLinkKinematics::doTMatrixToJointCoords(const TMatrix3<double> &tmatrix, ParallelogramLinkKinematics::coords_type &jointCoords, const ParallelogramLinkKinematics::coords_type *refJointCoordsPtr /*= NULL*/)
// be used in inverse kinematics
// 현재 index of axes가 0부터 시작되므로 base frame of a robot의 index는 -1이다
// tmatrix: the 4x4 transformation matrix of position & orientation of the end-effector of a robot
// jointCoords: result of inverse kinematics expressed by a joint coordinates, [ rad ]
// refJointCoordsPtr: in solving inverse kinematics, reference value to decide the resultant joint values of inverse kinematics
//  if a robot is of serial type, refCartesianCoordsPtr != NULL
//  if a robot is of in-parallel type  refCartesianCoordsPtr == NULL ( no need )
{
	const size_t nDOF = getDOF();
	if (nDOF != 6) return false;
	if (jointCoords.size() < nDOF) jointCoords.resize(nDOF);

	const double nx = tmatrix.X().x(), ny = tmatrix.X().y(), nz = tmatrix.X().z();
	const double tx = tmatrix.Y().x(), ty = tmatrix.Y().y(), tz = tmatrix.Y().z();
	const double bx = tmatrix.Z().x(), by = tmatrix.Z().y(), bz = tmatrix.Z().z();
	const double px = tmatrix.T().x(), py = tmatrix.T().y(), pz = tmatrix.T().z();
/*
	const double dUpperLimit[6] = {  160.0*MathConstant::D2R, 150.0*MathConstant::D2R,  135.0*MathConstant::D2R,  140.0*MathConstant::D2R,  120.0*MathConstant::D2R,  360.0*MathConstant::D2R };
	const double dLowerLimit[6] = { -160.0*MathConstant::D2R, -90.0*MathConstant::D2R, -150.0*MathConstant::D2R, -140.0*MathConstant::D2R, -120.0*MathConstant::D2R, -360.0*MathConstant::D2R };

	const double d1 = 0.475, d4 = 0.350, d6 = 0.095;
	const double a1 = 0.150, a2 = 0.350, a3 = 0.100;
*/
	const double d1 = getDHParam(0).getD(), d4 = getDHParam(3).getD(), d6 = getDHParam(5).getD();
	const double a1 = getDHParam(0).getA(), a2 = getDHParam(1).getA(), a3 = getDHParam(2).getA();

	const double L1 = std::sqrt(std::pow(px-d6*bx,2.0) + std::pow(py-d6*by,2.0)) - a1;
	const double L2 = pz - d6 * bz - d1;

	const double A0 = L1*L1 + L2*L2 + a3*a3 + d4*d4 - a2*a2;
	const double A1 = -2.0 * (L1*a3 - L2*d4);
	const double A2 = -2.0 * (L1*d4 + L2*a3);

	//  solve theta23 = theta2 + theta3
	const double dDiscriminent = A1*A1 + A2*A2 - A0*A0;
	if (dDiscriminent < 0.0) return false;

	double aRealRoot[2];
	aRealRoot[0] = (-A1 + std::sqrt(dDiscriminent)) / (A0 - A2);
	aRealRoot[1] = (-A1 - std::sqrt(dDiscriminent)) / (A0 - A2);
/*
#ifdef _DEBUG
	KcTRACE( "<<  The Solution of Inverse Kinematics expressed by Joint Coordinates, [ degree ]  >>\n" );
#endif  // _DEBUG
*/
	if (refJointCoordsPtr)
	{
		double dAngle1, dAngle2;
		double s1, c1, s23, c23;

		// theta1
		dAngle1 = std::atan2(py-d6*by, px-d6*bx);
		//dAngle2 = std::atan2(-(py-d6*by), -(px-d6*bx));
		dAngle2 = dAngle1 >= 0.0 ? dAngle1 - MathConstant::PI : dAngle1 + MathConstant::PI;
		jointCoords[0] = fabs(dAngle1-(*refJointCoordsPtr)[0]) < fabs(dAngle2-(*refJointCoordsPtr)[0]) ? dAngle1 : dAngle2;
		if (!checkJointLimit(0, jointCoords[0])) return false;
		s1 = std::sin(jointCoords[0]);
		c1 = std::cos(jointCoords[0]);

		// solve theta23 = theta2 + theta3
		const double aTheta23[2] = { 2.0*atan(aRealRoot[0]), 2.0*atan(aRealRoot[1]) };

		// theta2
		double aTmpTheta2[2];
		for (int i = 0 ; i < 2 ; ++i)
		{
			s23 = std::sin(aTheta23[i]);
			c23 = std::cos(aTheta23[i]);

			dAngle1 = std::atan2(L1-a3*s23-d4*c23, L2-a3*c23+d4*s23);
			//dAngle2 = std::atan2(-(L1-a3*s23-d4*c23), -(L2-a3*c23+d4*s23));
			dAngle2 = dAngle1 >= 0.0 ? dAngle1 - MathConstant::PI : dAngle1 + MathConstant::PI;
			aTmpTheta2[i] = fabs(dAngle1 - (*refJointCoordsPtr)[1]) < fabs(dAngle2 - (*refJointCoordsPtr)[1]) ? dAngle1 : dAngle2;
		}

		if (fabs(aTmpTheta2[0] - (*refJointCoordsPtr)[1]) < fabs(aTmpTheta2[1] - (*refJointCoordsPtr)[1]))
		{
			// theta2
			jointCoords[1] = aTmpTheta2[0];
			// theta3: DH notation 상에서의 3축의 각도, theta3 = theta23 - theta2 = theta3' - theta2
			jointCoords[2] = aTheta23[0] - jointCoords[1];
		}
		else
		{
			// theta2
			jointCoords[1] = aTmpTheta2[1];
			// theta3: DH notation 상에서의 3축의 각도, theta3 = theta23 - theta2 = theta3' - theta2
			jointCoords[2] = aTheta23[1] - jointCoords[1];
		}

		//if (!checkJointLimit(1, jointCoords[1]) || !checkJointLimit(2, jointCoords[2])) return false;
		if (!checkJointLimit(1, jointCoords[1]) || !checkJointLimit(2, jointCoords[1]+jointCoords[2])) return false;

		// rear link의 간섭에 의한 theta2 & theta3'의 각도 제한, -59.0 deg <= theta3' - theta2 <= 59.0 deg
		if (!checkLimit(jointCoords[2]-jointCoords[1], MathUtil::toRad(-59.0), MathUtil::toRad(59.0))) return false;

		// theta3': 3축 motor의 회전각, theta3' = theta2 + theta3
		const double s3 = std::sin(jointCoords[1]+jointCoords[2]), c3 = std::cos(jointCoords[1]+jointCoords[2]);

		// theta5
		dAngle1 = std::atan2(std::sqrt(std::pow(c1*c3*nx+s1*c3*ny-s3*nz,2.0)+std::pow(c1*c3*tx+s1*c3*ty-s3*tz,2.0)), c1*c3*bx+s1*c3*by-s3*bz);
		//dAngle2 = std::atan2(-std::sqrt(std::pow(c1*c3*nx+s1*c3*ny-s3*nz,2.0)+std::pow(c1*c3*tx+s1*c3*ty-s3*tz,2.0)), -(c1*c3*bx+s1*c3*by-s3*bz));
		//dAngle1 = std::atan2(-std::sqrt(std::pow(c1*c3*nx+s1*c3*ny-s3*nz,2.0)+std::pow(c1*c3*tx+s1*c3*ty-s3*tz,2.0)), c1*c3*bx+s1*c3*by-s3*bz);
		//dAngle2 = std::atan2(std::sqrt(std::pow(c1*c3*nx+s1*c3*ny-s3*nz,2.0)+std::pow(c1*c3*tx+s1*c3*ty-s3*tz,2.0)), -(c1*c3*bx+s1*c3*by-s3*bz));
		dAngle2 = dAngle1 >= 0.0 ? dAngle1 - MathConstant::PI : dAngle1 + MathConstant::PI;
		//dAngle2 = -dAngle1;
		jointCoords[4] = fabs(dAngle1 - (*refJointCoordsPtr)[4]) < fabs(dAngle2 - (*refJointCoordsPtr)[4]) ? dAngle1 : dAngle2;
		if (!checkJointLimit(4, jointCoords[4])) return false;

		// theta4 & theta6
		if (fabs(jointCoords[4]) > MathConstant::EPS)  // when theta5 != 0
		{
			// theta6
			dAngle1 = std::atan2(c1*c3*tx+s1*c3*ty-s3*tz, -(c1*c3*nx+s1*c3*ny-s3*nz));
			//dAngle2 = std::atan2(c1*c3*tx+s1*c3*ty-s3*tz, -(c1*c3*nx+s1*c3*ny-s3*nz));
			dAngle2 = dAngle1 >= 0.0 ? dAngle1 - MathConstant::PI : dAngle1 + MathConstant::PI;
			jointCoords[5] = fabs(dAngle1-(*refJointCoordsPtr)[5]) < fabs(dAngle2-(*refJointCoordsPtr)[5]) ? dAngle1 : dAngle2;
			if (!checkJointLimit(5, jointCoords[5])) return false;

			// theta4
			dAngle1 = std::atan2(-(s1*bx-c1*by), -(c1*s3*bx+s1*s3*by+c3*bz));
			//dAngle2 = std::atan2(s1*bx-c1*by, c1*s3*bx+s1*s3*by+c3*bz);
			dAngle2 = dAngle1 >= 0.0 ? dAngle1 - MathConstant::PI : dAngle1 + MathConstant::PI;
			jointCoords[3] = fabs(dAngle1-(*refJointCoordsPtr)[3]) < fabs(dAngle2-(*refJointCoordsPtr)[3]) ? dAngle1 : dAngle2;
			if (!checkJointLimit(3, jointCoords[3])) return false;
		}
		else  // when theta5 == 0
		{
			// theta46 = thteta4 + theta6
			dAngle1 = std::atan2(-(s1*nx-c1*ny), -(c1*s3*nx+s1*s3*ny+c3*nz));
			//dAngle2 = std::atan2(s1*nx-c1*ny, c1*s3*nx+s1*s3*ny+c3*nz);
			dAngle2 = dAngle1 >= 0.0 ? dAngle1 - MathConstant::PI : dAngle1 + MathConstant::PI;
			const double dTheta46 = fabs(dAngle1-(*refJointCoordsPtr)[3]-(*refJointCoordsPtr)[5]) < fabs(dAngle2-(*refJointCoordsPtr)[3]-(*refJointCoordsPtr)[5]) ? dAngle1 : dAngle2;

			const double dAxisRange3 = getJointParam(3).getUpperLimit() - getJointParam(3).getLowerLimit();
			const double dAxisRange5 = getJointParam(5).getUpperLimit() - getJointParam(5).getLowerLimit();

			jointCoords[3] = dTheta46 * dAxisRange3 / (dAxisRange3 + dAxisRange5);
			jointCoords[5] = dTheta46 * dAxisRange5 / (dAxisRange3 + dAxisRange5);

			if (!checkJointLimit(3, jointCoords[3]) || !checkJointLimit(5, jointCoords[5])) return false;
		}
	}
	else  // if refJointCoordsPtr == NULL
	{
		double s1, c1, s23, c23;
		double dAngle;

		// theta1
		dAngle = std::atan2(py-d6*by, px-d6*bx);
		if (!checkJointLimit(0, dAngle))
		{
			//dAngle = std::atan2(-(py-d6*by), -(px-d6*bx));
			dAngle = dAngle >= 0.0 ? dAngle - MathConstant::PI : dAngle + MathConstant::PI;
			if (!checkJointLimit(0, dAngle)) return false;
		}
		jointCoords[0] = dAngle;
		s1 = std::sin(jointCoords[0]);
		c1 = std::cos(jointCoords[0]);

		bool bHaveSolution = false;
		// theta23 = theta2 + theta3'
		for (int i = 0 ; i < 2 ; ++i)
		{
			const double dTheta23 = 2.0 * atan(aRealRoot[i]);
		
			s23 = std::sin(dTheta23);
			c23 = std::cos(dTheta23);

			// theta2
			dAngle = std::atan2(L1-a3*s23-d4*c23, L2-a3*c23+d4*s23);
			if (!checkJointLimit(1, dAngle))
			{
				//dAngle = std::atan2( -(L1-a3*s23-d4*c23), -(L2-a3*c23+d4*s23));
				dAngle = dAngle >= 0.0 ? dAngle - MathConstant::PI : dAngle + MathConstant::PI;
				if (!checkJointLimit(1, dAngle)) continue;
			}
			jointCoords[1] = dAngle;

			// theta3  :  DH notation 상에서의 3축의 각도, theta3 = theta23 - theta2 = theta3' - theta2
			jointCoords[2] = dTheta23 - jointCoords[1];
			//if (!checkJointLimit(2, jointCoords[2])) continue;
			if (!checkJointLimit(2, dTheta23)) continue;
			else
			{
				bHaveSolution = true;
				break;
			}
		}
		if (!bHaveSolution) return false;

		// rear link의 간섭에 의한 theta2 & theta3'의 각도 제한,  -59.0 deg <= theta3' - theta2 <= 59.0 deg
		if (!checkLimit(jointCoords[2]-jointCoords[1], MathUtil::toRad(-59.0), MathUtil::toRad(59.0))) return false;

		// theta3': 3축 motor의 회전각, theta3' = theta2 + theta3
		const double s3 = std::sin(jointCoords[1]+jointCoords[2]), c3 = std::cos( jointCoords[1]+jointCoords[2]);

		// theta5
		dAngle = std::atan2(std::sqrt(std::pow(c1*c3*nx+s1*c3*ny-s3*nz,2.0)+std::pow(c1*c3*tx+s1*c3*ty-s3*tz,2.0)), c1*c3*bx+s1*c3*by-s3*bz);
		//dAngle = std::atan2(-std::sqrt(std::pow(c1*c3*nx+s1*c3*ny-s3*nz,2.0)+std::pow(c1*c3*tx+s1*c3*ty-s3*tz,2.0)), c1*c3*bx+s1*c3*by-s3*bz);
		if (!checkJointLimit(4, dAngle))
		{
			//dAngle = std::atan2(-std::sqrt(std::pow(c1*c3*nx+s1*c3*ny-s3*nz,2.0)+std::pow(c1*c3*tx+s1*c3*ty-s3*tz,2.0)), -(c1*c3*bx+s1*c3*by-s3*bz));
			//dAngle = std::atan2(std::sqrt(std::pow(c1*c3*nx+s1*c3*ny-s3*nz,2.0)+std::pow(c1*c3*tx+s1*c3*ty-s3*tz,2.0)), -(c1*c3*bx+s1*c3*by-s3*bz));
			dAngle = dAngle >= 0.0 ? dAngle - MathConstant::PI : dAngle + MathConstant::PI;
			//dAngle = -dAngle;
			if (!checkJointLimit(4, dAngle)) return false;
		}
		jointCoords[4] = dAngle;

		// theta4 & theta6
		if (fabs(jointCoords[4]) > MathConstant::EPS)  // when theta5 != 0
		{
			// theta6
			dAngle = std::atan2(c1*c3*tx+s1*c3*ty-s3*tz, -(c1*c3*nx+s1*c3*ny-s3*nz));
			if (!checkJointLimit(5, dAngle))
			{
				//dAngle = std::atan2( c1*c3*tx+s1*c3*ty-s3*tz, -(c1*c3*nx+s1*c3*ny-s3*nz));
				dAngle = dAngle >= 0.0 ? dAngle - MathConstant::PI : dAngle + MathConstant::PI;
				if (checkJointLimit(5, dAngle)) return false;
			}
			jointCoords[5] = dAngle;

			// theta4
			dAngle = std::atan2(-(s1*bx-c1*by), -(c1*s3*bx+s1*s3*by+c3*bz));
			if (!checkJointLimit(3, dAngle)) {
				//dAngle = std::atan2(s1*bx-c1*by, c1*s3*bx+s1*s3*by+c3*bz);
				dAngle = dAngle >= 0.0 ? dAngle - MathConstant::PI : dAngle + MathConstant::PI;
				if (!checkJointLimit(3, dAngle)) return false;
			}
			jointCoords[3] = dAngle;
		}
		else  // when theta5 == 0
		{
			const double dAxisRange3 = getJointParam(3).getUpperLimit() - getJointParam(3).getLowerLimit();
			const double dAxisRange5 = getJointParam(5).getUpperLimit() - getJointParam(5).getLowerLimit();

			// theta46 = thteta4 + theta6
			dAngle = std::atan2(-(s1*nx-c1*ny), -(c1*s3*nx+s1*s3*ny+c3*nz));

			jointCoords[3] = dAngle * dAxisRange3 / (dAxisRange3 + dAxisRange5);
			jointCoords[5] = dAngle * dAxisRange5 / (dAxisRange3 + dAxisRange5);
			if (!checkJointLimit(3, jointCoords[3]) || !checkJointLimit(5, jointCoords[5]))
			{
				//dAngle = std::atan2(s1*nx-c1*ny, c1*s3*nx+s1*s3*ny+c3*nz);
				dAngle = dAngle >= 0.0 ? dAngle - MathConstant::PI : dAngle + MathConstant::PI;

				jointCoords[3] = dAngle * dAxisRange3 / (dAxisRange3 + dAxisRange5);
				jointCoords[5] = dAngle * dAxisRange5 / (dAxisRange3 + dAxisRange5);
				if (!checkJointLimit(3, jointCoords[3]) || !checkJointLimit(5, jointCoords[5])) return false;
			}
		}
	}

	// DH notation 상에서의 3축의 각도, theta3는 실제 3축 motor의 회전각, theta3'와 2축 motor의 회전각, theta2'의 차임
	// 즉, theta3 = theta3' - theta2임.
	// 따라서, 3축의 motor 회전각, theta3' = theta3 + theta2
	// 여기서, theta2' = theta2임
	jointCoords[2] += jointCoords[1];
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

TMatrix3<double> ParallelogramLinkKinematics::doCalcRMatrix(const double alpha, const double beta, const double gamma)
// be used in inverse kinematics
// fixed angle, X(alpha) => Y(beta) => Z(gamma): R = Rz(gamma) * Ry(beta) * Rx(alpba) from Craig's Book
{
	return base_type::doCalcRMatrix(alpha, beta, gamma);
}

bool ParallelogramLinkKinematics::doCalcRotationAngle(const TMatrix3<double> &tmatrix, double &alpha, double &beta, double &gamma)
// be used in forward kinematics
// fixed angle, X(alpha) => Y(beta) => Z(gamma): R = Rz(gamma) * Ry(beta) * Rx(alpba) from Craig's Book
{
	return base_type::doCalcRotationAngle(tmatrix, alpha, beta, gamma);
}

}  // namespace swl
