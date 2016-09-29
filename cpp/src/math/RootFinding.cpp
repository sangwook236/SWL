#include "swl/Config.h"
#include "swl/math/RootFinding.h"
#include "swl/math/MathUtil.h"
#include "swl/base/LogException.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//-----------------------------------------------------------------------------------------
// Root Finding.

double RootFinding::secant(double init, double (*func)(double), double tolerance /*= MathConstant::EPS*/)
{
	double delta, final;
	double front, rear;
	
	final = init + 2.0;
	front = (*func)(init);
	rear = (*func)(final);
	
	int i = 0;
	do
	{
		if (MathUtil::isZero(rear-front, tolerance))
		{
			throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
			//return 0.0;
		}

		delta = -rear * (final - init) / (rear - front);
		init = final;
		final += delta;
		front = rear;
		rear = (*func)(final);
		++i;

		if (i >= MathConstant::ITERATION_LIMIT)
		{
			throw LogException(LogException::L_ERROR, "iteration overflow", __FILE__, __LINE__, __FUNCTION__);
			//return final;
		}
	} while (fabs(delta) > tolerance || fabs(rear) > tolerance);
	
	return final;
}

double RootFinding::bisection(double left, double right, double (*func)(double), double tolerance /*= MathConstant::EPS*/)
{
	double front, rear;
	front = (*func)(left);
	rear = (*func)(right);

	if (front * rear >= 0.0)
	{
		throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
		//return 0.0;
	}

	double mid, delta, temp;
	int i = 0;
	do
	{
		mid = (left + right) / 2.0;
		temp = (*func)(mid);
		
		if (temp * front > 0.0)
		{
			front = temp;
			delta = mid - left;
			left = mid;
		}
		else
		{
			rear = temp;
			delta = mid - right;
			right = mid;
		}
		++i;

		if (i >= MathConstant::ITERATION_LIMIT)
		{
			throw LogException(LogException::L_ERROR, "iteration overflow", __FILE__, __LINE__, __FUNCTION__);
			//return mid;
		}
	} while (fabs(delta) > tolerance || fabs(temp) > tolerance);
	
	return mid;
}

double RootFinding::falsePosition(double left, double right, double (*func)(double), double tolerance /*= MathConstant::EPS*/)
{
	double front, rear;
	front = (*func)(left);
	rear = (*func)(right);
	
	if (front * rear >= 0.0)
	{
		throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
		//return 0.0;
	}

	double inter, delta, temp;
	int i = 0;
	do
	{
		inter =  (right * front - left * rear) / (front - rear);
		temp = (*func)(inter);
		if (temp * front > 0.0)
		{
			front = temp;
			delta = inter - left;
			left = inter;
		}
		else
		{
			rear = temp;
			delta = inter - right;
			right = inter;
		}
		++i;

		if (i >= MathConstant::ITERATION_LIMIT)
		{
			throw LogException(LogException::L_ERROR, "iteration overflow", __FILE__, __LINE__, __FUNCTION__);
			//return inter;
		}
	}  while (fabs(delta) > tolerance || fabs(temp) > tolerance);
	
	return inter;
}

bool RootFinding::quadratic(const double coeffArr[3], Complex<double> rootArr[2], double tolerance /*= MathConstant::EPS*/)
// 2nd order polynomial
// coefficients are arranged by a descending order
{
	if (MathUtil::isZero(coeffArr[0], tolerance))
	{
		throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
		//return false;
	}

	// calculate normalized coefficients
	const double u = coeffArr[1] / coeffArr[0];
	const double v = coeffArr[2] / coeffArr[0];

	//
	const double dDet = u * u - 4.0 * v;
	if (dDet >= 0.0)
	{
		if (u >= 0.0)
		{
			rootArr[0].real() = -2.0*v / (u + ::sqrt(dDet));
			rootArr[1].real() = -(u + ::sqrt(dDet)) / 2.0;
			rootArr[0].imag() = rootArr[1].imag() = 0.0;
		}
		else
		{
			rootArr[0].real() = (-u + ::sqrt(dDet)) / 2.0;
			rootArr[1].real() = -2.0*v / (u - ::sqrt(dDet));
			rootArr[0].imag() = rootArr[1].imag() = 0.0;
		}
	}
	else
	{
		rootArr[0].real() = rootArr[1].real() = -u / 2.0;
		rootArr[0].imag() = ::sqrt(-dDet) / 2.0;
		rootArr[1].imag() = -rootArr[0].imag();
	}

	return true;
}

bool RootFinding::quadratic(const std::vector<double>& coeffCtr, std::vector<Complex<double> >& rootCtr, double tolerance /*= MathConstant::EPS*/)
// 2nd order polynomial
// coefficients are arranged by a descending order
{
	if (coeffCtr.size() != 3)
	{
		throw LogException(LogException::L_ERROR, "illegal dimension", __FILE__, __LINE__, __FUNCTION__);
		//return false;
	}
	else if (MathUtil::isZero(coeffCtr[0], tolerance))
	{
		throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
		//return false;
	}

	// calculate normalized coefficients
	const double u = coeffCtr[1] / coeffCtr[0];
	const double v = coeffCtr[2] / coeffCtr[0];

	//
	const double dDet = u * u - 4.0 * v;
	Complex<double> aRoot;
	if (dDet >= 0.0)  // two roots are real
	{
		aRoot.imag() = 0.0;
		if (u >= 0.0)
		{
			aRoot.real() = -2.0*v / (u + ::sqrt(dDet));
			rootCtr.push_back(aRoot);
			aRoot.real() = -(u + ::sqrt(dDet)) / 2.0;
			rootCtr.push_back(aRoot);
		}
		else
		{
			aRoot.real() = (-u + ::sqrt(dDet)) / 2.0;
			rootCtr.push_back(aRoot);
			aRoot.real() = -2.0*v / (u - ::sqrt(dDet));
			rootCtr.push_back(aRoot);
		}
	}
	else  // two roots are imaginary
	{
		aRoot.real() = -u / 2.0;
		aRoot.imag() = ::sqrt(-dDet) / 2.0;
		rootCtr.push_back(aRoot);
		aRoot.imag() = -aRoot.imag();
		rootCtr.push_back(aRoot);
	}

	return true;
}

bool RootFinding::cubic(const double coeffArr[4], Complex<double> rootArr[3], double tolerance /*= MathConstant::EPS*/)
// 3rd order polynomial
// coefficients are arranged by a descending order
{
	if (-tolerance <= coeffArr[0] && coeffArr[0] <= tolerance)
	{
		throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
		//return false;
	}

	// calculate normalized coefficients
	const double a = coeffArr[1] / coeffArr[0];
	const double b = coeffArr[2] / coeffArr[0];
	const double c = coeffArr[3] / coeffArr[0];

	const double p = a*a / 9.0 - b / 3.0;
	const double q = -a*a*a / 27.0 + a*b / 6.0 - c / 2.0;
	const double dDet = q*q - p*p*p;
	
	if (dDet > 0.0)  // one root is real, two roots are imaginary
	{
		// contain the case where p == 0
		double t1 = q+::sqrt(dDet), t2 = q-::sqrt(dDet);
		t1 = t1 >= 0.0 ? ::pow(t1, 1.0/3.0) : -::pow(-t1, 1.0/3.0);
		t2 = t2 >= 0.0 ? ::pow(t2, 1.0/3.0) : -::pow(-t2, 1.0/3.0);
		rootArr[0].real() = t1 + t2 - a / 3.0;
		rootArr[0].imag() = 0.0;
		rootArr[1].real() = rootArr[2].real() = -(t1 + t2) / 2.0 - a / 3.0;
		rootArr[1].imag() = (t1 - t2) * ::sqrt(3.0) / 2.0;
		rootArr[2].imag() = -rootArr[1].imag();
	}
	else if (fabs(dDet) < tolerance)  // three roots are real, two roots are equal
	{
		const double qq = q >= 0.0 ? ::pow(q, 1.0/3.0) : -::pow(-q, 1.0/3.0);
		rootArr[0].real() = 2.0 * qq - a / 3.0;
		rootArr[1].real() = rootArr[2].real() = -qq - a / 3.0;
		rootArr[0].imag() = rootArr[1].imag() = rootArr[2].imag() = 0.0;
	}
	else  // three roots are real and distinct
	{
		if (p < 0.0)
		{
			throw LogException(LogException::L_ERROR, "domain error", __FILE__, __LINE__, __FUNCTION__);
			//return false;
		}

		const double u = ::acos(q / ::pow(p, 1.5));
		rootArr[0].real() = 2.0 * ::sqrt(p) * ::cos(u / 3.0) - a / 3.0;
		rootArr[1].real() = 2.0 * ::sqrt(p) * ::cos((u + MathConstant::_2_PI) / 3.0) - a / 3.0;
		rootArr[2].real() = 2.0 * ::sqrt(p) * ::cos((u + MathConstant::_4_PI) / 3.0) - a / 3.0;
		rootArr[0].imag() = rootArr[1].imag() = rootArr[2].imag() = 0.0;
	}

	return true;
}

bool RootFinding::cubic(const std::vector<double>& coeffCtr, std::vector<Complex<double> >& rootCtr, double tolerance /*= MathConstant::EPS*/)
// 3rd order polynomial
// coefficients are arranged by a descending order
{
	if (coeffCtr.size() != 4)
	{
		throw LogException(LogException::L_ERROR, "illegal dimension", __FILE__, __LINE__, __FUNCTION__);
		//return false;
	}
	else if (MathUtil::isZero(coeffCtr[0], tolerance))
	{
		throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
		//return false;
	}

	// calculate normalized coefficients
	const double a = coeffCtr[1] / coeffCtr[0];
	const double b = coeffCtr[2] / coeffCtr[0];
	const double c = coeffCtr[3] / coeffCtr[0];

	const double p = a*a / 9.0 - b / 3.0;
	const double q = -a*a*a / 27.0 + a*b / 6.0 - c / 2.0;
	const double dDet = q*q - p*p*p;
	
	Complex<double> aRoot;
	if (dDet > 0.0)  // one root is real, two roots are imaginary
	{
		// contain the case where p == 0
		double t1 = q+::sqrt(dDet), t2 = q-::sqrt(dDet);
		t1 = t1 >= 0.0 ? ::pow(t1, 1.0/3.0) : -::pow(-t1, 1.0/3.0);
		t2 = t2 >= 0.0 ? ::pow(t2, 1.0/3.0) : -::pow(-t2, 1.0/3.0);
		aRoot.real() = t1 + t2 - a / 3.0;
		aRoot.imag() = 0.0;
		rootCtr.push_back(aRoot);
		aRoot.real() = -(t1 + t2) / 2.0 - a / 3.0;
		aRoot.imag() = (t1 - t2) * ::sqrt(3.0) / 2.0;
		rootCtr.push_back(aRoot);
		aRoot.imag() = -aRoot.imag();
		rootCtr.push_back(aRoot);
	}
	else if (fabs(dDet) < tolerance)  // three roots are real, two roots are equal
	{
		const double qq = q >= 0.0 ? ::pow(q, 1.0/3.0) : -::pow(-q, 1.0/3.0);
		aRoot.imag() = 0.0;
		aRoot.real() = 2.0 * qq - a / 3.0;
		rootCtr.push_back(aRoot);
		aRoot.real() = -qq - a / 3.0;
		rootCtr.push_back(aRoot);
		rootCtr.push_back(aRoot);
	}
	else  // three roots are real and distinct
	{
		if (p < 0.0)
		{
			throw LogException(LogException::L_ERROR, "domain error", __FILE__, __LINE__, __FUNCTION__);
			//return false;
		}

		const double u = ::acos(q / ::pow(p, 1.5));
		aRoot.imag() = 0.0;
		aRoot.real() = 2.0 * ::sqrt(p) * ::cos(u / 3.0) - a / 3.0;
		rootCtr.push_back(aRoot);
		aRoot.real() = 2.0 * ::sqrt(p) * ::cos((u + MathConstant::_2_PI) / 3.0) - a / 3.0;
		rootCtr.push_back(aRoot);
		aRoot.real() = 2.0 * ::sqrt(p) * ::cos((u + MathConstant::_4_PI) / 3.0) - a / 3.0;
		rootCtr.push_back(aRoot);
	}

	return true;
}

bool RootFinding::quartic(const double coeffArr[5], Complex<double> rootArr[4], double tolerance /*= MathConstant::EPS*/)
// 4th order polynomial
// coefficients are arranged by a descending order
{
	if (MathUtil::isZero(coeffArr[0], tolerance))
	{
		throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
		//return false;
	}

	// calculate normalized coefficients
	const double a1 = coeffArr[1] / coeffArr[0];
	const double a2 = coeffArr[2] / coeffArr[0];
	const double a3 = coeffArr[3] / coeffArr[0];
	const double a4 = coeffArr[4] / coeffArr[0];
	
	const double t = -a1 / 4.0;
	const double b1 = 6.0 * t*t + 3.0 * a1 * t + a2;
	const double b2 = 4.0 * t*t*t + 3.0 * a1 * t*t + 2.0 * a2 * t + a3;
	const double b3 = t*t*t*t + a1 * t*t*t + a2 * t*t + a3 * t + a4;
	
	const double arCoeff1[4] = { 1.0, 2.0*b1, b1*b1-4.0*b3, -b2*b2 };
	Complex<double> arRoot1[3];

	cubic(arCoeff1, arRoot1);
	
	for (int i = 0 ; i < 3 ; ++i)
	{
		if (fabs(arRoot1[i].imag()) <= tolerance)  // nearly real root
		{
			if (arRoot1[i].real() > 0.0)
			{
				const double p = ::sqrt(arRoot1[i].real());
				const double q1 = ((b1 + p*p) - b2 / p) * 0.5;
				const double q2 = ((b1 + p*p) + b2 / p) * 0.5;

				double arCoeff2[3] = { 1.0, p, q1 };
				quadratic(arCoeff2, &rootArr[0]);
				
				arCoeff2[1] = -p;
				arCoeff2[2] = q2;
				quadratic(arCoeff2, &rootArr[2]);

				for (int j = 0 ; j < 4 ; ++j) rootArr[j].real() += t;
			}
		}
	}

	return true;
}

bool RootFinding::quartic(const std::vector<double>& coeffCtr, std::vector<Complex<double> >& rootCtr, double tolerance /*= MathConstant::EPS*/)
// 4th order polynomial
// coefficients are arranged by a descending order
{
	if (coeffCtr.size() != 5)
	{
		throw LogException(LogException::L_ERROR, "illegal dimension", __FILE__, __LINE__, __FUNCTION__);
		//return false;
	}
	else if (MathUtil::isZero(coeffCtr[0], tolerance))
	{
		throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
		//return false;
	}

	// calculate normalized coefficients
	const double a1 = coeffCtr[1] / coeffCtr[0];
	const double a2 = coeffCtr[2] / coeffCtr[0];
	const double a3 = coeffCtr[3] / coeffCtr[0];
	const double a4 = coeffCtr[4] / coeffCtr[0];
	
	const double t = -a1 / 4.0;
	const double b1 = 6.0 * t*t + 3.0 * a1 * t + a2;
	const double b2 = 4.0 * t*t*t + 3.0 * a1 * t*t + 2.0 * a2 * t + a3;
	const double b3 = t*t*t*t + a1 * t*t*t + a2 * t*t + a3 * t + a4;
	
	std::vector<double> vtrCoeff1(4);
	vtrCoeff1[0] = 1.0;
	vtrCoeff1[1] = 2.0 * b1;
	vtrCoeff1[2] = b1*b1 - 4.0*b3;
	vtrCoeff1[3] = -b2 * b2;
	std::vector<Complex<double> > vtrRoot1;

	cubic(vtrCoeff1, vtrRoot1);
	if (vtrRoot1.size() != 3)
	{
		throw LogException(LogException::L_ERROR, "illegal dimension", __FILE__, __LINE__, __FUNCTION__);
		//return false;
	}
	
	for (int i = 0 ; i < 3 ; ++i)
	{
		if (fabs(vtrRoot1[i].imag()) <= tolerance)  // nearly real root
		{
			if (vtrRoot1[i].real() > 0.0)
			{
				std::vector<double> vtrCoeff2(3);
				double p = ::sqrt(vtrRoot1[i].real());

				vtrCoeff2[0] = 1.0;
				vtrCoeff2[1] = p;
				vtrCoeff2[2] = ((b1 + p*p) - b2 / p) * 0.5;
				quadratic(vtrCoeff2, rootCtr);
				
				vtrCoeff2[1] = -p;
				vtrCoeff2[2] = ((b1 + p*p) + b2 / p) * 0.5;
				quadratic(vtrCoeff2, rootCtr);

				const std::vector<Complex<double> >::size_type nSize = rootCtr.size();
				if (nSize < 4)
				{
					throw LogException(LogException::L_ERROR, "illegal dimension", __FILE__, __LINE__, __FUNCTION__);
					//return false;
				}
				std::vector<Complex<double> >::size_type j = nSize;
				do
				{
					--j;
					rootCtr[j].real() += t;
				} while (j > nSize-4 && j != 0);
			}
		}
	}
	return true;
}

static void solveQuadraticForBairstowMethod(double u, double v, std::vector<Complex<double> >& rootCtr)
// solve x^2 + u*x + v = 0
{
	const double dDet = u * u - 4.0 * v;
	
	Complex<double> aRoot;
	if (dDet >= 0.0)  // two roots are real
	{
		aRoot.imag() = 0.0;
		if (u >= 0.0)
		{
			aRoot.real() = -2.0*v / (u + ::sqrt(dDet));
			rootCtr.push_back(aRoot);
			aRoot.real() = -(u + ::sqrt(dDet)) / 2.0;
			rootCtr.push_back(aRoot);
		}
		else
		{
			aRoot.real() = (-u + ::sqrt(dDet)) / 2.0;
			rootCtr.push_back(aRoot);
			aRoot.real() = -2.0*v / (u - ::sqrt(dDet));
			rootCtr.push_back(aRoot);
		}
	}
	else  // two roots are imaginary
	{
		aRoot.real() = -u / 2.0;
		aRoot.imag() = ::sqrt(-dDet) / 2.0;
		rootCtr.push_back(aRoot);
		aRoot.imag() = -aRoot.imag();
		rootCtr.push_back(aRoot);
	}
}

bool RootFinding::bairstow(const std::vector<double>& coeffCtr, std::vector<Complex<double> >& rootCtr, double tolerance /*= MathConstant::EPS*/)
// n-th order polynomial
// coeffCtr are arranged by a descending order
{
	std::vector<double>::size_type nOrder = coeffCtr.size() - 1;
	if (nOrder < 1)
	{
		throw LogException(LogException::L_ERROR, "illegal dimension", __FILE__, __LINE__, __FUNCTION__);
		//return false;
	}
	else if (MathUtil::isZero(coeffCtr[0], tolerance))
	{
		throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
		//return false;
	}

	std::vector<double> vtrCoeffA(coeffCtr), vtrCoeffB(nOrder + 1), vtrCoeffC(nOrder + 1);
	std::vector<double>::size_type i, j;

	// calculate normalized coefficients
	i = nOrder + 1;
	do
	{
		--i;
		vtrCoeffA[i] /= vtrCoeffA[0];
	} while (i > 0);
	
	double u = 0.0, v = 0.0, du = 1.0, dv = 1.0;
	j = nOrder;
	int iIteration, iCount;
	while (j > 2)
	{
		iIteration = iCount = 0;
		do
		{
			i = 1;
			vtrCoeffB[0] = vtrCoeffC[0] = 1.0;
			vtrCoeffB[1] = vtrCoeffA[1] - u;
			vtrCoeffC[1] = vtrCoeffB[1] - u;
			do
			{
				++i;
				vtrCoeffB[i] = vtrCoeffA[i] - u*(vtrCoeffB[i-1]) - v*(vtrCoeffB[i-2]);
				vtrCoeffC[i] = vtrCoeffB[i] - u*(vtrCoeffC[i-1]) - v*(vtrCoeffC[i-2]);
			} while (i < j);
			
			const double dDet = vtrCoeffC[j-1] * vtrCoeffC[j-3] - vtrCoeffC[j-2] * vtrCoeffC[j-2];
			if (MathUtil::isZero(dDet))
			{
				u += 0.5;
				v += 0.5;
				continue;
			}
			else
			{
				du = (vtrCoeffB[j] * vtrCoeffC[j-3] - vtrCoeffB[j-1] * vtrCoeffC[j-2]) / dDet;
				dv = (vtrCoeffB[j-1] * vtrCoeffC[j-1] - vtrCoeffB[j] * vtrCoeffC[j-2]) / dDet;
			}
			u += du;
			v += dv;
			
			++iCount;
			if (iCount > MathConstant::ITERATION_LIMIT)
			{
				iCount = 0;
				u += 1.0;
				v += 1.0;

				++iIteration;
				if (iIteration >= MathConstant::ITERATION_LIMIT)
				{
					throw LogException(LogException::L_ERROR, "iteration overflow", __FILE__, __LINE__, __FUNCTION__);
					//break;
				}
				else continue;
			}
		} while ((fabs(du) + fabs(dv)) > tolerance);
		
		solveQuadraticForBairstowMethod(u, v, rootCtr);
		
		for (i = 0 ; i <= j - 2 ; ++i) vtrCoeffA[i] = vtrCoeffB[i];
		j -= 2;
	}
	
	if (2 == j)  
		solveQuadraticForBairstowMethod(vtrCoeffA[1], vtrCoeffA[2], rootCtr);
	else
		rootCtr.push_back(Complex<double>(-vtrCoeffA[1], 0.0));

	return true;
}

}  // namespace swl
