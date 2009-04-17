#if !defined(__SWL_VIEW__VIEW_MATRIX__H_)
#define __SWL_VIEW__VIEW_MATRIX__H_ 1


namespace swl {

//--------------------------------------------------------------------------
// 2D homogeneous transformation matrix

class ViewMatrix2
{
public:
	//typedef ViewMatrix2 base_type;
	typedef double value_type;

public:
	ViewMatrix2();
	explicit ViewMatrix2(const value_type mat[6]);
	explicit ViewMatrix2(const value_type x[2], const value_type y[2], const value_type t[2]);
	ViewMatrix2(const ViewMatrix2& rhs);

	ViewMatrix2& operator=(const ViewMatrix2& rhs);

public:
	///
	void set(const value_type mat[6]);
	void get(value_type mat[6]) const;

	///
	void translate(const value_type dx, const value_type dy);
	void rotate(const value_type rad);
	void scale(const value_type sx, const value_type sy);
	void identity();

	///
	ViewMatrix2 operator*(const ViewMatrix2& rhs) const;

	///
	bool transformPoint(const value_type x, const value_type y, value_type& xt, value_type& yt) const;
	bool inverseTransformPoint(const value_type x, const value_type y, value_type& xt, value_type& yt, const value_type tol = 1.0e-7f) const;

private:
	/// 2D homogeneous transformation matrix
	/// [ m0 m1 m2 ] = [ x1 y1 t1 ]
	/// [ m3 m4 m5 ] = [ x2 y2 t2 ]
	/// [  0  0  1 ] = [  0  0  1 ]
	value_type tmat_[6];
};


//--------------------------------------------------------------------------
//  3D homogeneous transformation matrix

class ViewMatrix3
{
public:
	//typedef ViewMatrix3 base_type;
	typedef double value_type;

public:
	ViewMatrix3();
	explicit ViewMatrix3(const value_type mat[12]);
	explicit ViewMatrix3(const value_type x[3], const value_type y[3], const value_type z[3], const value_type t[3]);
	ViewMatrix3(const ViewMatrix3& rhs);

	ViewMatrix3& operator=(const ViewMatrix3& rhs);

public:
	///
	void set(const value_type mat[12]);
	void get(value_type mat[12]) const;

	///
	void translate(const value_type dx, const value_type dy, const value_type dz);
	void rotateX(const value_type rad);
	void rotateY(const value_type rad);
	void rotateZ(const value_type rad);
	void scale(const value_type sx, const value_type sy, const value_type sz);
	void identity();

	///
	ViewMatrix3 operator*(const ViewMatrix3& rhs) const;

	///
	bool transformPoint(const value_type x, const value_type y, const value_type z, value_type& xt, value_type& yt, value_type& zt) const;
	bool inverseTransformPoint(const value_type x, const value_type y, const value_type z, value_type& xt, value_type& yt, value_type& zt, const value_type tol = 1.0e-7f) const;

private:
	/// 3D homogeneous transformation matrix
	/// [ m0 m1  m2  m3 ] = [ x1 y1 z1 t1 ]
	/// [ m4 m5  m6  m7 ]   [ x2 y2 z2 t2 ]
	/// [ m8 m9 m10 m11 ]   [ x3 y3 z3 t3 ]
	/// [  0  0   0   1 ]   [  0  0  0  1 ]
	value_type tmat_[12];
};

}  // namespace swl


#endif  // __SWL_VIEW__VIEW_MATRIX__H_
