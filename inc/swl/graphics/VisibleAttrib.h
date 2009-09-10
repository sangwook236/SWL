#if !defined(__SWL_GRAPHICS__VISIBLE_ATTRIBUTE__H_)
#define __SWL_GRAPHICS__VISIBLE_ATTRIBUTE__H_ 1


namespace swl {

//-----------------------------------------------------------------------------------------
// struct VisibleAttrib

struct VisibleAttrib
{
public:
	enum ViewMode { WIRE_FRAME, FLAT_SHADING, SMOOTH_SHADING };

public:
	VisibleAttrib(const bool isVisible = true, const ViewMode viewMode = SMOOTH_SHADING)
	: isVisible_(isVisible), viewMode_(viewMode)
	{}
	VisibleAttrib(const VisibleAttrib &rhs)
	: isVisible_(rhs.isVisible_), viewMode_(rhs.viewMode_)
	{}
	~VisibleAttrib()
	{}

	VisibleAttrib & operator=(const VisibleAttrib &rhs)
	{
		if (this == &rhs) return *this;
		//static_cast<base_type &>(*this) = rhs;
		isVisible_ = rhs.isVisible_;
		viewMode_ = rhs.viewMode_;
		return *this;
	}

public:
	/// mutator
	void setVisible(bool isVisible)  {  isVisible_ = isVisible;  }
	/// accessor
	bool isVisible() const  {  return isVisible_;  }

	/// mutator
	void setViewMode(const ViewMode viewMode)  {  viewMode_ = viewMode;  }
	/// accessor
	ViewMode getViewMode() const  {  return viewMode_;  }

private:
	bool isVisible_;

	ViewMode viewMode_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__VISIBLE_ATTRIBUTE__H_
