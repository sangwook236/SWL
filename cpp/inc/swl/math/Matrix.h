#if !defined(__SWL_MATH__MATRIX__H_)
#define __SWL_MATH__MATRIX__H_ 1


#include <valarray>
#include <cmath>


namespace swl {
	
//-----------------------------------------------------------------------------------------
// class Matrix
		
template<typename T>
class Matrix
{
public:
    typedef T value_type;

public:
	Matrix(const int row = 1, const int col = 1)
	: row_(row), col_(col), entry_()
	{}
	Matrix(const Matrix &rhs)
	: row_(rhs.row_), col_(rhs.col_), entry_(rhs.entry_)
	{}
	~Matrix() {}
	
	Matrix & operator=(const Matrix &rhs)
	{
		if (this == &rhs) return *this;
		row_ = rhs.row_;
		col_ = rhs.col_;
		entry_ = rhs.entry_;
		return *this;
	}

private:
	///
	size_t row_;
	size_t col_;

	/// entry
	std::valarray<T> entry_;
};


//-----------------------------------------------------------------------------------------
// Matrix API
		

}  // namespace swl


#endif  // __SWL_MATH__MATRIX__H_
