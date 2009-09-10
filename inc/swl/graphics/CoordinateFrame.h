#if !defined(__SWL_GRAPHICS__COORDINATE_FRAME__H_)
#define __SWL_GRAPHICS__COORDINATE_FRAME__H_ 1


#include "swl/graphics/GraphicsObj.h"
#include "swl/math/TMatrix.h"


namespace swl {

//-----------------------------------------------------------------------------------------
//

#if defined(_MSC_VER)
#pragma warning(disable:4231)
SWL_GRAPHICS_EXPORT_TEMPLATE template class SWL_GRAPHICS_API TVector3<double>;
SWL_GRAPHICS_EXPORT_TEMPLATE template class SWL_GRAPHICS_API TMatrix3<double>;
#endif


//-----------------------------------------------------------------------------------------
// class CoordinateFrame

class SWL_GRAPHICS_API CoordinateFrame: public GraphicsObj
{
public:
	typedef GraphicsObj			base_type;
	typedef TMatrix3<double>	frame_type;

public:
#if defined(_UNICODE) || defined(UNICODE)
	CoordinateFrame(const std::wstring &name = std::wstring());
#else
	CoordinateFrame(const std::string &name = std::string());
#endif
	CoordinateFrame(const CoordinateFrame &rhs);
	virtual ~CoordinateFrame();

	CoordinateFrame& operator=(const CoordinateFrame &rhs);

public:
	/// frame name
#if defined(_UNICODE) || defined(UNICODE)
	void setName(const std::wstring& name)  {  name_ = name;  }
#else
	void setName(const std::string& name)  {  name_ = name;  }
#endif
#if defined(_UNICODE) || defined(UNICODE)
	const std::wstring& getName() const  {  return name_;  }
#else
	const std::string& getName() const  {  return name_;  }
#endif

private:
	/// frame name
#if defined(_UNICODE) || defined(UNICODE)
	std::wstring name_;
#else
	std::string name_;
#endif

	///
	frame_type frame_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__COORDINATE_FRAME__H_
