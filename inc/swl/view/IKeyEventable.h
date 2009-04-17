#if !defined(__SWL_VIEW__KEY_EVENTABLE_INTERFACE__H_)
#define __SWL_VIEW__KEY_EVENTABLE_INTERFACE__H_ 1


namespace swl {

//-----------------------------------------------------------------------------------
// 

struct IKeyEventable
{
public:
	//typedef IKeyEventable base_type;

protected:
	IKeyEventable()  {}
public:
	virtual ~IKeyEventable()  {}

private:
	IKeyEventable(const IKeyEventable&);
	IKeyEventable& operator=(const IKeyEventable&);

public:
	virtual void pressKey(const int key) = 0;
	virtual void releaseKey(const int key) = 0;
};

}  // namespace swl


#endif  // __SWL_VIEW__KEY_EVENTABLE_INTERFACE__H_
