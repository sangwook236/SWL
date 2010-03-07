#if !defined(__SWL_GRAPHICS__VISIBLE_ATTRIBUTE__H_)
#define __SWL_GRAPHICS__VISIBLE_ATTRIBUTE__H_ 1


namespace swl {

//-----------------------------------------------------------------------------------------
// struct VisibleAttrib

struct VisibleAttrib
{
public:
	enum PolygonMode { WIRE_FRAME, FLAT_SHADING, SMOOTH_SHADING };

public:
	VisibleAttrib(const bool isVisible = true, const PolygonMode viewMode = SMOOTH_SHADING)
	: isVisible_(isVisible), polygonMode_(viewMode)
	{}
	VisibleAttrib(const VisibleAttrib &rhs)
	: isVisible_(rhs.isVisible_), polygonMode_(rhs.polygonMode_)
	{}
	~VisibleAttrib()
	{}

	VisibleAttrib & operator=(const VisibleAttrib &rhs)
	{
		if (this == &rhs) return *this;
		//static_cast<base_type &>(*this) = rhs;
		isVisible_ = rhs.isVisible_;
		polygonMode_ = rhs.polygonMode_;
		return *this;
	}

public:
	/// mutator
	void setVisible(bool isVisible)  {  isVisible_ = isVisible;  }
	/// accessor
	bool isVisible() const  {  return isVisible_;  }

	/// mutator
	void setPolygonMode(const PolygonMode polygonMode)  {  polygonMode_ = polygonMode;  }
	/// accessor
	PolygonMode getPolygonMode() const  {  return polygonMode_;  }

private:
	bool isVisible_;

	PolygonMode polygonMode_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__VISIBLE_ATTRIBUTE__H_
