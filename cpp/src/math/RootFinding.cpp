#include "swl/Config.h"
#include "swl/math/RootFinding.h"
#include "swl/math/MathUtil.h"
#include "swl/base/LogException.h"
#include <gsl/gsl_poly.h>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//-----------------------------------------------------------------------------------------
// Root Finding.

double RootFinding::secant(double init, double (*func)(double), double tol /*= MathConstant::EPS*/)
{
	double delta, final;
	double front, rear;
	
	final = init + 2.0;
	front = (*func)(init);
	rear = (*func)(final);
	
	int i = 0;
	do
	{
		if (MathUtil::isZero(rear-front, tol))
		{
			throw LogException(LogException::L_ERROR, "Divide by zero", __FILE__, __LINE__, __FUNCTION__);
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
			throw LogException(LogException::L_ERROR, "Iteration overflow", __FILE__, __LINE__, __FUNCTION__);
			//return final;
		}
	} while (std::abs(delta) > tol || std::abs(rear) > tol);
	
	return final;
}

double RootFinding::bisection(double left, double right, double (*func)(double), double tol /*= MathConstant::EPS*/)
{
	double front, rear;
	front = (*func)(left);
	rear = (*func)(right);

	if (front * rear >= 0.0)
	{
		throw LogException(LogException::L_ERROR, "Illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
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
			throw LogException(LogException::L_ERROR, "Iteration overflow", __FILE__, __LINE__, __FUNCTION__);
			//return mid;
		}
	} while (std::abs(delta) > tol || std::abs(temp) > tol);
	
	return mid;
}

double RootFinding::falsePosition(double left, double right, double (*func)(double), double tol /*= MathConstant::EPS*/)
{
	double front, rear;
	front = (*func)(left);
	rear = (*func)(right);
	
	if (front * rear >= 0.0)
	{
		throw LogException(LogException::L_ERROR, "Illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
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
			throw LogException(LogException::L_ERROR, "Iteration overflow", __FILE__, __LINE__, __FUNCTION__);
			//return inter;
		}
	}  while (std::abs(delta) > tol || std::abs(temp) > tol);
	
	return inter;
}

bool RootFinding::quadratic(const std::array<double, 3>& coeffs, std::array<std::complex<double>, 2>& roots, double tol /*= MathConstant::EPS*/)
// 2nd order polynomial.
// Coefficients are arranged by a descending order.
{
	if (MathUtil::isZero(coeffs[0], tol))
	{
		throw LogException(LogException::L_ERROR, "Illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
		//return false;
	}

	// Calculate normalized coefficients.
	const double u = coeffs[1] / coeffs[0];
	const double v = coeffs[2] / coeffs[0];

	//
	const double dDet = u * u - 4.0 * v;
	if (dDet >= 0.0)
	{
		if (u >= 0.0)
		{
			const double val = u + std::sqrt(dDet);
			roots[0].real(-2.0*v / val);
			roots[0].imag(0.0);
			roots[1].real(-val * 0.5);
			roots[1].imag(0.0);
		}
		else
		{
			const double val = u - std::sqrt(dDet);
			roots[0].real(-val * 0.5);
			roots[0].imag(0.0);
			roots[1].real(-2.0*v / val);
			roots[1].imag(0.0);
		}
	}
	else
	{
		roots[0].real(-u * 0.5);
		roots[0].imag(std::sqrt(-dDet) * 0.5);
		roots[1].real(roots[0].real());
		roots[1].imag(-roots[0].imag());
	}

	return true;
}

bool RootFinding::cubic(const std::array<double, 4>& coeffs, std::array<std::complex<double>, 3>& roots, double tol /*= MathConstant::EPS*/)
// 3rd order polynomial.
// Coefficients are arranged by a descending order.
{
	if (-tol <= coeffs[0] && coeffs[0] <= tol)
	{
		throw LogException(LogException::L_ERROR, "Illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
		//return false;
	}

	// Calculate normalized coefficients.
	const double a = coeffs[1] / coeffs[0];
	const double b = coeffs[2] / coeffs[0];
	const double c = coeffs[3] / coeffs[0];

	const double p = a*a / 9.0 - b / 3.0;
	const double q = -a*a*a / 27.0 + a*b / 6.0 - c / 2.0;
	const double dDet = q*q - p*p*p;
	
	if (dDet > 0.0)  // One root is real, two roots are imaginary.
	{
		// Contain the case where p == 0.
		double t1 = q+std::sqrt(dDet), t2 = q-std::sqrt(dDet);
		t1 = t1 >= 0.0 ? std::pow(t1, 1.0/3.0) : -std::pow(-t1, 1.0/3.0);
		t2 = t2 >= 0.0 ? std::pow(t2, 1.0/3.0) : -std::pow(-t2, 1.0/3.0);
		roots[0].real(t1 + t2 - a / 3.0);
		roots[0].imag(0.0);
		roots[1].real(-(t1 + t2) / 2.0 - a / 3.0);
		roots[1].imag((t1 - t2) * std::sqrt(3.0) / 2.0);
		roots[2].real(roots[1].real());
		roots[2].imag(-roots[1].imag());
	}
	else if (std::abs(dDet) < tol)  // Three roots are real, two roots are equal.
	{
		const double qq = q >= 0.0 ? std::pow(q, 1.0/3.0) : -std::pow(-q, 1.0/3.0);
		roots[0].real(2.0 * qq - a / 3.0);
		roots[0].imag(0.0);
		roots[1].real(-qq - a / 3.0);
		roots[1].imag(0.0);
		roots[2].real(roots[1].real());
		roots[2].imag(0.0);
	}
	else  // Three roots are real and distinct.
	{
		if (p < 0.0)
		{
			throw LogException(LogException::L_ERROR, "Domain error", __FILE__, __LINE__, __FUNCTION__);
			//return false;
		}

		const double u = ::acos(q / std::pow(p, 1.5));
		const double val1 = 2.0 * std::sqrt(p), val2 = a / 3.0;
		roots[0].real(val1 * std::cos(u / 3.0) - val2);
		roots[1].real(val1 * std::cos((u + MathConstant::_2_PI) / 3.0) - val2);
		roots[2].real(val1 * std::cos((u + MathConstant::_4_PI) / 3.0) - val2);
		roots[0].imag(0.0);
		roots[1].imag(0.0);
		roots[2].imag(0.0);
	}

	return true;
}

bool RootFinding::quartic(const std::array<double, 5>& coeffs, std::array<std::complex<double>, 4>& roots, double tol /*= MathConstant::EPS*/)
// 4th order polynomial.
// Coefficients are arranged by a descending order.
{
	if (MathUtil::isZero(coeffs[0], tol))
	{
		throw LogException(LogException::L_ERROR, "Illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
		//return false;
	}

	// Calculate normalized coefficients.
	const double a1 = coeffs[1] / coeffs[0];
	const double a2 = coeffs[2] / coeffs[0];
	const double a3 = coeffs[3] / coeffs[0];
	const double a4 = coeffs[4] / coeffs[0];
	
	const double t = -a1 * 0.25;
	const double b1 = 6.0 * t*t + 3.0 * a1 * t + a2;
	const double b2 = 4.0 * t*t*t + 3.0 * a1 * t*t + 2.0 * a2 * t + a3;
	const double b3 = t*t*t*t + a1 * t*t*t + a2 * t*t + a3 * t + a4;
	
	const std::array<double, 4> coeffs1 = { 1.0, 2.0*b1, b1*b1-4.0*b3, -b2*b2 };
	std::array<std::complex<double>, 3> arRoot1;

	cubic(coeffs1, arRoot1);
	
	for (int i = 0 ; i < 3 ; ++i)
	{
		if (std::abs(arRoot1[i].imag()) <= tol)  // Nearly real root.
		{
			if (arRoot1[i].real() > 0.0)
			{
				const double p = std::sqrt(arRoot1[i].real());
				const double q1 = ((b1 + p*p) - b2 / p) * 0.5;
				const double q2 = ((b1 + p*p) + b2 / p) * 0.5;

				std::array<double, 3> coeffs2 = { 1.0, p, q1 };
				std::array<std::complex<double>, 2> roots2;
				quadratic(coeffs2, roots2);
				roots[0] = roots2[0];
				roots[1] = roots2[1];
				
				coeffs2[1] = -p;
				coeffs2[2] = q2;
				quadratic(coeffs2, roots2);
				roots[2] = roots2[0];
				roots[3] = roots2[1];

				for (int j = 0 ; j < 4 ; ++j) roots[j].real(roots[j].real() + t);
			}
		}
	}

	return true;
}

static void solveQuadraticForBairstowMethod(double u, double v, std::vector<std::complex<double> >& roots)
// Solve x^2 + u*x + v = 0.
{
	const double dDet = u * u - 4.0 * v;
	
	std::complex<double> aRoot;
	if (dDet >= 0.0)  // Two roots are real.
	{
		aRoot.imag(0.0);
		if (u >= 0.0)
		{
			const double val = u + std::sqrt(dDet);
			aRoot.real(-2.0*v / val);
			roots.push_back(aRoot);
			aRoot.real(-val / 2.0);
			roots.push_back(aRoot);
		}
		else
		{
			const double val = -u + std::sqrt(dDet);
			aRoot.real(val * 0.5);
			roots.push_back(aRoot);
			aRoot.real(2.0*v / val);
			roots.push_back(aRoot);
		}
	}
	else  // Two roots are imaginary.
	{
		aRoot.real(-u * 0.5);
		aRoot.imag(std::sqrt(-dDet) * 0.5);
		roots.push_back(aRoot);
		aRoot.imag(-aRoot.imag());
		roots.push_back(aRoot);
	}
}

bool RootFinding::bairstow(const std::vector<double>& coeffs, std::vector<std::complex<double> >& roots, double tol /*= MathConstant::EPS*/)
// n-th order polynomial.
// Coeffs are arranged by a descending order.
{
	std::vector<double>::size_type nOrder = coeffs.size() - 1;
	if (nOrder < 1)
	{
		throw LogException(LogException::L_ERROR, "illegal dimension", __FILE__, __LINE__, __FUNCTION__);
		//return false;
	}
	else if (MathUtil::isZero(coeffs[0], tol))
	{
		throw LogException(LogException::L_ERROR, "Illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
		//return false;
	}

	std::vector<double> vtrCoeffA(coeffs), vtrCoeffB(nOrder + 1), vtrCoeffC(nOrder + 1);
	std::vector<double>::size_type i, j;

	// Calculate normalized coefficients.
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
					throw LogException(LogException::L_ERROR, "Iteration overflow", __FILE__, __LINE__, __FUNCTION__);
					//break;
				}
				else continue;
			}
		} while ((std::abs(du) + std::abs(dv)) > tol);
		
		solveQuadraticForBairstowMethod(u, v, roots);
		
		for (i = 0 ; i <= j - 2 ; ++i) vtrCoeffA[i] = vtrCoeffB[i];
		j -= 2;
	}
	
	if (2 == j)  
		solveQuadraticForBairstowMethod(vtrCoeffA[1], vtrCoeffA[2], roots);
	else
		roots.push_back(std::complex<double>(-vtrCoeffA[1], 0.0));

	return true;
}

/*static*/ size_t RootFinding::solveSystemOfQuadraticEquations(const double a1, const double b1, const double c1, const double d1, const double a2, const double b2, const double c2, const double d2, double &x0, double& x1, double tol /*= MathConstant::TOL_5*/)
{
	const double eps = std::numeric_limits<double>::epsilon();

#if 1
	const double a = a1 - a2;
	const double b = b1 - b2;
	const double c = c1 - c2;
	const double d = d1 - d2;
	if (std::abs(a) <= eps && std::abs(b) <= 0 && std::abs(c) <= eps)
	{
		// Infinitely many real solutions.
		//	- Two identical quadratic equations.
		//	- Two identical linear equations.
		//		Two identical vertical lines.
		if (std::abs(d) <= eps) return (size_t)-1;
		// No real solution (when they never intersect).
		//	- Two (parallel) quadratic equations: y = x^2 - 3*x + 2, y = x^2 - 3*x + 3.
		//	- A quadratic equation and a linear equation: y = x^2 + 1, y = x.
		//	- Two parallel linear equations: vertical, horizontal, inclined lines.
		else return 0;
	}
	else if (std::abs(c) <= eps)  // y can have arbitrary values.
	{
		// Infinitely many real solutions.
		//	- Solutions are one vertical line: x = x0 when a = 0.
		//	- Solutions are two vertical lines: x = x0 and x = x1 when a != 0.
		gsl_poly_solve_quadratic(a, b, d, &x0, &x1);
		return (size_t)-1;
	}
	else
	{
		// One real solution.
		//	- Two quadratic curves touch each other.
		//	- A linear equation just touches a quadratic equation.
		//	- Two lines intersect at a point.
		// Two real solutions.
		//	- A quadratic equation and a linear equation.
		//	- Two quadratic equations.
		return gsl_poly_solve_quadratic(a1 * c2 - a2 * c1, b1 * c2 - b2 * c1, d1 * c2 - d2 * c1, &x0, &x1);
	}
#else
	if (std::abs(c1) > eps && std::abs(c2) > eps)
		return gsl_poly_solve_quadratic(a1 * c2 - a2 * c1, b1 * c2 - b2 * c1, d1 * c2 - d2 * c1, &x0, &x1);
	else if (std::abs(c1) > eps)  // c2 is nearly zero.
		return gsl_poly_solve_quadratic(a2, b2, d2, &x0, &x1);
	else if (std::abs(c2) > eps)  // c1 is nearly zero.
		return gsl_poly_solve_quadratic(a1, b1, d1, &x0, &x1);
	else  // Both c1 and c2 are nearly zero.
	{
		double s0 = 0.0, s1 = 0.0, t0 = 0.0, t1 = 0.0;
		const int num1 = gsl_poly_solve_quadratic(a1, b1, d1, &s0, &s1);
		const int num2 = gsl_poly_solve_quadratic(a2, b2, d2, &t0, &t1);

		if (num1 <= 0 || num2 <= 0) return 0;
		else if (1 == num1)
		{
			if (1 == num2)
			{
				if (std::abs(s0 - t0) <= tol)
				{
					x0 = 0.5 * (s0 + t0);
					return 1;
				}
				else return 0;
			}
			else if (2 == num2)
			{
				if (std::abs(s0 - t0) <= tol)
				{
					x0 = 0.5 * (s0 + t0);
					return 1;
				}
				else if (std::abs(s0 - t1) <= tol)
				{
					x0 = 0.5 * (s0 + t1);
					return 1;
				}
				else return 0;
			}
			else assert(false);
		}
		else if (2 == num1)
		{
			if (1 == num2)
			{
				if (std::abs(s0 - t0) <= tol)
				{
					x0 = 0.5 * (s0 + t0);
					return 1;
				}
				else if (std::abs(s1 - t0) <= tol)
				{
					x0 = 0.5 * (s1 + t0);
					return 1;
				}
				else return 0;
			}
			else if (2 == num2)
			{
				const std::vector<double> dists({ std::abs(s0 - t0), std::abs(s0 - t1), std::abs(s1 - t0), std::abs(s1 - t1) });
				const auto minIt = std::min_element(dists.begin(), dists.end());
				const auto minIdx = std::distance(dists.begin(), minIt);
				if (*minIt > tol) return 0;
				else
				{
					switch (minIdx)
					{
					case 0:
						x0 = 0.5 * (s0 + t0);
						if (dists[3] <= tol)
						{
							x1 = 0.5 * (s1 + t1);
							return 2;
						}
						else return 1;
					case 1:
						x0 = 0.5 * (s0 + t1);
						if (dists[2] <= tol)
						{
							x1 = 0.5 * (s1 + t0);
							return 2;
						}
						else return 1;
					case 2:
						x0 = 0.5 * (s1 + t0);
						if (dists[1] <= tol)
						{
							x1 = 0.5 * (s0 + t1);
							return 2;
						}
						else return 1;
					case 3:
						x0 = 0.5 * (s1 + t1);
						if (dists[0] <= tol)
						{
							x1 = 0.5 * (s0 + t0);
							return 2;
						}
						else return 1;
					default:
						assert(false);
						break;
					}
				}
			}
			else assert(false);
		}
		else assert(false);
	}
#endif
}

}  // namespace swl
