#if !defined(__SWL_OGL_VIEW_TEST__VIEW_EVENT_HANDLER__H_)
#define __SWL_OGL_VIEW_TEST__VIEW_EVENT_HANDLER__H_ 1


namespace swl {

struct MouseEvent;
struct KeyEvent;


//-----------------------------------------------------------------------------------
// 

struct MousePressHandler
{
public:
	//typedef MousePressHandler base_type;

public:
	void operator()(const MouseEvent &evt) const;
};

//-----------------------------------------------------------------------------------
// 

struct MouseReleaseHandler
{
public:
	//typedef MouseReleaseHandler base_type;

public:
	void operator()(const MouseEvent &evt) const;
};

//-----------------------------------------------------------------------------------
// 

struct MouseMoveHandler
{
public:
	//typedef MouseMoveHandler base_type;

public:
	void operator()(const MouseEvent &evt) const;
};

//-----------------------------------------------------------------------------------
// 

struct MouseWheelHandler
{
public:
	//typedef MouseWheelHandler base_type;

public:
	void operator()(const MouseEvent &evt) const;
};

//-----------------------------------------------------------------------------------
// 

struct MouseClickHandler
{
public:
	//typedef MouseClickHandler base_type;

public:
	void operator()(const MouseEvent &evt) const;
};

//-----------------------------------------------------------------------------------
// 

struct MouseDoubleClickHandler
{
public:
	//typedef MouseDoubleClickHandler base_type;

public:
	void operator()(const MouseEvent &evt) const;
};

//-----------------------------------------------------------------------------------
// 

struct KeyPressHandler
{
public:
	//typedef KeyPressHandler base_type;

public:
	void operator()(const KeyEvent &evt) const;
};

//-----------------------------------------------------------------------------------
// 

struct KeyReleaseHandler
{
public:
	//typedef KeyReleaseHandler base_type;

public:
	void operator()(const KeyEvent &evt) const;
};

//-----------------------------------------------------------------------------------
// 

struct KeyHitHandler
{
public:
	//typedef KeyHitHandler base_type;

public:
	void operator()(const KeyEvent &evt) const;
};

}  // namespace swl


#endif  // __SWL_OGL_VIEW_TEST__VIEW_EVENT_HANDLER__H_
