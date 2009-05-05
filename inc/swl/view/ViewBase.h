#if !defined(__SWL_VIEW__VIEW_BASE__H_)
#define __SWL_VIEW__VIEW_BASE__H_ 1


namespace swl {

//-----------------------------------------------------------------------------------
// 

struct ViewBase
{
public:
	//typedef ViewBase base_type;

public:
	virtual ~ViewBase()  {}

public:
	virtual bool raiseDrawEvent(const bool isContextActivated) = 0;

	virtual bool initializeView() = 0;
	virtual bool resizeView(const int x1, const int y1, const int x2, const int y2) = 0;
};

}  // namespace swl


#endif  // __SWL_VIEW__VIEW_BASE__H_
