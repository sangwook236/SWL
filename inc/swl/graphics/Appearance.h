#if !defined(__SWL_GRAPHICS__APPEARANCE__H_)
#define __SWL_GRAPHICS__APPEARANCE__H_ 1


#include "swl/graphics/ExportGraphics.h"
#include "swl/graphics/Color.h"
#include "swl/graphics/TransparentAttrib.h"


namespace swl {

#if defined(_MSC_VER)
#pragma warning(disable:4231)
SWL_GRAPHICS_EXPORT_TEMPLATE template struct SWL_GRAPHICS_API RGBAColor<double>;
#endif


//-----------------------------------------------------------------------------------------
// class Appearance

class SWL_GRAPHICS_API Appearance
{
public:
	//typedef Appearance base_type;

public:
	Appearance();
	Appearance(const Appearance &rhs);
	virtual ~Appearance();

	Appearance& operator=(const Appearance &rhs);

public:
	/// accessor & mutator
	double & red()  {  return color_.r;  }
	double red() const  {  return color_.r;  }
	double & green()  {  return color_.g;  }
	double green() const  {  return color_.g;  }
	double & blue()  {  return color_.b;  }
	double blue() const  {  return color_.b;  }
	double & alpha()  {  return color_.a;  }
	double alpha() const  {  return color_.a;  }

	void setTransparent(bool isTransparent)  {  transparent_.setTransparent(isTransparent);  }
	bool isTransparent() const  {  return transparent_.isTransparent();  }

private:
	///
	RGBAColor<double> color_;
	TransparentAttrib transparent_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__APPEARANCE__H_
