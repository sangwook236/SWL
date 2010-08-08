#if !defined(__SWL_RND_UTIL_TEST__ADIS_USBZ__H_)
#define __SWL_RND_UTIL_TEST__ADIS_USBZ__H_ 1


// Internal file : BaseProcessing/AdisUsbz.h
#include <windows.h>

class AdisUsbz
{
public:
    AdisUsbz();
    virtual ~AdisUsbz();

#if defined(UNICODE) || defined(_UNICODE)
    bool Initialize( const wchar_t * aDeviceName );
#else
	bool Initialize( const char * aDeviceName );
#endif

	short ReadInt14( unsigned char aAddress );

private:
	short ReadInt16( unsigned char aAddress );

    void WriteLoopDelay ();
    void WriteStaticBits();

	void Rx(       void * aOut, unsigned char aOutSize );
	void Tx( const void * aIn , unsigned char aInSize  );

    void DeviceControl( unsigned int aIoCtl, const void * aIn, unsigned int aInSize, void * aOut, unsigned int aOutSize );

private:
    HANDLE mHandle;
};


#endif  // __SWL_RND_UTIL_TEST__ADIS_USBZ__H_
