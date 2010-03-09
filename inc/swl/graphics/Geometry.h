#if !defined(__SWL_GRAPHICS__GEOMETRY__H_)
#define __SWL_GRAPHICS__GEOMETRY__H_ 1


#include "swl/graphics/ExportGraphics.h"


namespace swl {

//-----------------------------------------------------------------------------------------
// class Geometry

class SWL_GRAPHICS_API Geometry
{
public:
	//typedef Geometry base_type;

public:
	Geometry();
	Geometry(const Geometry &rhs);
	virtual ~Geometry();

	Geometry & operator=(const Geometry &rhs);

public:
	virtual void draw() const = 0;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__GEOMETRY__H_
