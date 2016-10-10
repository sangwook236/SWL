#include "swl/Config.h"
#include "swl/math/RootFinding.h"
#include "swl/math/MathUtil.h"
#include "swl/base/LogException.h"
#include <gsl/gsl_poly.h>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cassert>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

// Solve x^2 + u*x + v = 0.
void solveQuadraticForBairstowMethod(double u, double v, std::vector<std::complex<double> >& roots)
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

}  // namespace local
}  // unnamed namespace

namespace swl {

//-----------------------------------------------------------------------------------------
// Root Finding.

double RootFinding::secant(double init, double (*func)(double), const double& tol /*= MathConstant::EPS*/)
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

double RootFinding::bisection(double left, double right, double (*func)(double), const double& tol /*= MathConstant::EPS*/)
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

double RootFinding::falsePosition(double left, double right, double (*func)(double), const double& tol /*= MathConstant::EPS*/)
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

bool RootFinding::quadratic(const std::array<double, 3>& coeffs, std::array<std::complex<double>, 2>& roots, const double& tol /*= MathConstant::EPS*/)
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

bool RootFinding::cubic(const std::array<double, 4>& coeffs, std::array<std::complex<double>, 3>& roots, const double& tol /*= MathConstant::EPS*/)
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

bool RootFinding::quartic(const std::array<double, 5>& coeffs, std::array<std::complex<double>, 4>& roots, const double& tol /*= MathConstant::EPS*/)
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

bool RootFinding::bairstow(const std::vector<double>& coeffs, std::vector<std::complex<double> >& roots, const double& tol /*= MathConstant::EPS*/)
{
	std::vector<double>::size_type nOrder = coeffs.size() - 1;
	if (nOrder < 1)
	{
		throw LogException(LogException::L_ERROR, "Illegal dimension", __FILE__, __LINE__, __FUNCTION__);
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
		
		local::solveQuadraticForBairstowMethod(u, v, roots);
		
		for (i = 0 ; i <= j - 2 ; ++i) vtrCoeffA[i] = vtrCoeffB[i];
		j -= 2;
	}
	
	if (2 == j)  
		local::solveQuadraticForBairstowMethod(vtrCoeffA[1], vtrCoeffA[2], roots);
	else
		roots.push_back(std::complex<double>(-vtrCoeffA[1], 0.0));

	return true;
}

/*static*/ size_t RootFinding::solveSystemOfQuadraticEquations(const std::array<double, 4>& coeffs1, const std::array<double, 4>& coeffs2, std::array<double, 2>& roots, const double& eps /*= MathConstant::EPS*/)
{
	const bool isVertical1 = std::abs(coeffs1[2]) <= eps;
	const bool isVertical2 = std::abs(coeffs2[2]) <= eps;

#if 1
	// It is important whether the ratios between corresponding coefficients, but not their differences, are the same or not.
	//	e.g.) a1 / a2 = b1 / b2 = c1 / c2 = d1 / d2 = ?
#if 0
	const double a = coeffs1[0] - coeffs2[0];
	const double b = coeffs1[1] - coeffs2[1];
	const double c = coeffs1[2] - coeffs2[2];
	const double d = coeffs1[3] - coeffs2[3];
#else
	// L1 norm.
	const double norm1 = std::accumulate(coeffs1.begin(), coeffs1.end(), 0.0, [](const double sum, const double elem) { return sum + std::abs(elem); });
	const double norm2 = std::accumulate(coeffs2.begin(), coeffs2.end(), 0.0, [](const double sum, const double elem) { return sum + std::abs(elem); });
	// L2 norm.
	//const double norm1 = std::sqrt(std::accumulate(coeffs1.begin(), coeffs1.end(), 0.0, [](const double sum, const double elem) { return sum + elem*elem; }));
	//const double norm2 = std::sqrt(std::accumulate(coeffs2.begin(), coeffs2.end(), 0.0, [](const double sum, const double elem) { return sum + elem*elem; }));
	// Squared L2 norm => (cannot use).
	//const double norm1 = std::accumulate(coeffs1.begin(), coeffs1.end(), 0.0, [](const double sum, const double elem) { return sum + elem*elem; });
	//const double norm2 = std::accumulate(coeffs2.begin(), coeffs2.end(), 0.0, [](const double sum, const double elem) { return sum + elem*elem; });
	// L-inf norm.
	//const double norm1 = std::sqrt(std::accumulate(coeffs1.begin(), coeffs1.end(), 0.0, [](const double maxElem, const double elem) { return std::max(maxElem, std::abs(elem)); }));
	//const double norm2 = std::sqrt(std::accumulate(coeffs2.begin(), coeffs2.end(), 0.0, [](const double maxElem, const double elem) { return std::max(maxElem, std::abs(elem)); }));
	if (norm1 <= eps || norm2 <= eps)
	{
		throw std::runtime_error("All the coefficients are nearly zero");
		return (size_t)-1;
	}

	std::array<double, 4> coeffs;
	std::transform(coeffs1.begin(), coeffs1.end(), coeffs2.begin(), coeffs.begin(), [&norm1, &norm2](const double coeff1, const double coeff2) { return coeff1 / norm1 - coeff2 / norm2; });

	const double& a = coeffs[0];
	const double& b = coeffs[1];
	const double& c = coeffs[2];
	const double& d = coeffs[3];
#endif

	if (std::abs(a) <= eps && std::abs(b) <= 0 && std::abs(c) <= eps)
	{
		// If d == 0, infinitely many real solutions exist.
		//	- Two identical quadratic equations.
		//	- Two identical linear equations.
		//		Two identical vertical lines.
		// Otherwise, no real solution (when they never intersect) exists.
		//	- Two (parallel) quadratic equations: y = x^2 - 3*x + 2, y = x^2 - 3*x + 3.
		//	- A quadratic equation and a linear equation: y = x^2 + 1, y = x.
		//	- Two parallel linear equations: vertical, horizontal, inclined lines.
		return std::abs(d) <= eps ? (size_t)-1 : 0;
	}
	// When either coeffs1[2] or coeffs2[2] is zero, y in the corresponding equation can be 'arbitrary' and x has one (when a = 0) or two (when a != 0) values.
	//	- This case is different from when c = 0, which means a case that y values are the same.
	else if (isVertical1 && isVertical2)
	{
		// We may have up to 4 vertical lines, where two of them are from the 1st equation and the other two are from the 2nd one.
		double x0 = 0.0, x1 = 0.0;
		switch (gsl_poly_solve_quadratic(coeffs1[0], coeffs1[1], coeffs1[3], &x0, &x1))
		{
		case 0:
			return 0;
		case 1:
			// Check if a vertical line is identical.
			if ((coeffs2[0] * x0 * x0 + coeffs2[1] * x0 + coeffs2[3]) <= eps)
			{
				roots[0] = x0;
				return (size_t)-1;
			}
			else return 0;
		case 2:
		{
			// Check if two vertical lines are identical.
			int numRoots = 0;
			if ((coeffs2[0] * x0 * x0 + coeffs2[1] * x0 + coeffs2[3]) <= eps)
			{
				roots[0] = x0;
				++numRoots;
			}
			if ((coeffs2[0] * x1 * x1 + coeffs2[1] * x1 + coeffs2[3]) <= eps)
			{
				if (numRoots > 0) roots[1] = x1;
				else roots[0] = x1;
				++numRoots;
			}
			return numRoots > 0 ? (size_t)-1 : 0;
		}
		default:
			assert(false);
			return 0;
		}
	}
	else if (isVertical1)
	{
		// The 1st equation has infinitely many real solutions.
		//	- Solutions are one vertical line: x = roots[0] when a = 0.
		//	- Solutions are two vertical lines: x = roots[0] and x = roots[1] when a != 0.
		// The final solutions are intersections of the 2nd equation with these vertical lines, so x values are decided by the 1st equation.
		return gsl_poly_solve_quadratic(coeffs1[0], coeffs1[1], coeffs1[3], &roots[0], &roots[1]);
	}
	else if (isVertical2)
	{
		// The 2nd equation has infinitely many real solutions.
		//	- Solutions are one vertical line: x = roots[0] when a = 0.
		//	- Solutions are two vertical lines: x = roots[0] and x = roots[1] when a != 0.
		// The final solutions are intersections of the 1st equation with these vertical lines, so x values are decided by the 2nd equation.
		return gsl_poly_solve_quadratic(coeffs2[0], coeffs2[1], coeffs2[3], &roots[0], &roots[1]);
	}
	else
	{
		// One real solution exists.
		//	- Two quadratic curves touch each other.
		//	- A linear equation just touches a quadratic equation.
		//	- Two lines intersect at a point.
		// Two real solutions exist.
		//	- A quadratic equation and a linear equation.
		//	- Two quadratic equations.
		return gsl_poly_solve_quadratic(coeffs1[0] * coeffs2[2] - coeffs2[0] * coeffs1[2], coeffs1[1] * coeffs2[2] - coeffs2[1] * coeffs1[2], coeffs1[3] * coeffs2[2] - coeffs2[3] * coeffs1[2], &roots[0], &roots[1]);
	}
#else
	if (!isVertical1 && !isVertical2)
		return gsl_poly_solve_quadratic(coeffs1[0] * coeffs2[2] - coeffs2[0] * coeffs1[2], coeffs1[1] * coeffs2[2] - coeffs2[1] * coeffs1[2], coeffs1[3] * coeffs2[2] - coeffs2[3] * coeffs1[2], &roots[0], &roots[1]);
	else if (!isVertical1)  // coeffs2[2] is nearly zero.
		return gsl_poly_solve_quadratic(coeffs2[0], coeffs2[1], coeffs2[3], &roots[0], &roots[1]);
	else if (!isVertical2)  // coeffs1[2] is nearly zero.
		return gsl_poly_solve_quadratic(coeffs1[0], coeffs1[1], coeffs1[3], &roots[0], &roots[1]);
	else  // Both coeffs1[2] and coeffs2[2] are nearly zero.
	{
		double x0 = 0.0, x1 = 0.0;
		switch (gsl_poly_solve_quadratic(coeffs1[0], coeffs1[1], coeffs1[3], &x0, &x1))
		{
		case 0:
			return 0;
		case 1:
			// Check if a vertical line is identical.
			if ((coeffs2[0] * x0 * x0 + coeffs2[1] * x0 + coeffs2[3]) <= eps)
			{
				roots[0] = x0;
				return (size_t)-1;
			}
			else return 0;
		case 2:
		{
			// Check if two vertical lines are identical.
			int numRoots = 0;
			if ((coeffs2[0] * x0 * x0 + coeffs2[1] * x0 + coeffs2[3]) <= eps)
			{
				roots[0] = x0;
				++numRoots;
			}
			if ((coeffs2[0] * x1 * x1 + coeffs2[1] * x1 + coeffs2[3]) <= eps)
			{
				if (numRoots > 0) roots[1] = x1;
				else roots[0] = x1;
				++numRoots;
			}
			return numRoots > 0 ? (size_t)-1 : 0;
		}
		default:
			assert(false);
			return 0;
		}
	}
#endif
}

}  // namespace swl
