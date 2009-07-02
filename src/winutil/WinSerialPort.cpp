#include "swl/winutil/WinSerialPort.h"
#include "swl/util/GuardedBuffer.h"
#include <vector>
#include <iostream>

#if defined(_MSC_VER) && defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//-----------------------------------------------------------------------------------
//	serial port

WinSerialPort::WinSerialPort()
: hComPort_(INVALID_HANDLE_VALUE)
{
	memset(&ovSend_, 0, sizeof(OVERLAPPED));
	memset(&ovRecv_, 0, sizeof(OVERLAPPED));
	ovSend_.hEvent = ::CreateEvent(
		NULL,   // default security attributes 
		FALSE,  // auto reset event 
		FALSE,  // not signaled 
		NULL    // no name
	);
	ovRecv_.hEvent = ::CreateEvent(
		NULL,
		FALSE,
		FALSE,
		NULL
	);
}

WinSerialPort::~WinSerialPort()
{
	disconnect();

	if (NULL != ovSend_.hEvent)
	{
		::CloseHandle(ovSend_.hEvent);
		ovSend_.hEvent = NULL;
	}
	if (NULL != ovRecv_.hEvent)
	{
		::CloseHandle(ovRecv_.hEvent);
		ovRecv_.hEvent = NULL;
	}
}

#if defined(_UNICODE) || defined(__UNICODE)
bool WinSerialPort::connect(const wchar_t* portName, const int baudRate, const size_t inQueueSize, const size_t outQueueSize)
#else
bool WinSerialPort::connect(const char *portName, const int baudRate, const size_t inQueueSize, const size_t outQueueSize)
#endif
{
	hComPort_ = CreateFile(
		portName,
		GENERIC_READ | GENERIC_WRITE,
		0,    // exclusive access 
		NULL, // default security attributes 
		OPEN_EXISTING,
		FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
		NULL 
	);
	if (INVALID_HANDLE_VALUE == hComPort_)
	{
		// Handle the error. 
		std::cerr << "CreateFile failed with error, " << GetLastError() << ": Port:" << portName << ", BaudRate:" << baudRate << ", Parity:None, Stop bits:1" << std::endl;
		//::CloseHandle(hComPort_);
		return false;
	}

	// Set the event mask. 
	//if (!SetCommMask(hComPort_, EV_CTS | EV_DSR))
	if (!SetCommMask(hComPort_, EV_RXCHAR))
	{
		// Handle the error. 
		std::cerr << "SetCommMask failed with error: " << GetLastError() << std::endl;
		return false;
	}
	
	// set sizes of inqueue & outqueue
	SetupComm(hComPort_, (DWORD)inQueueSize, (DWORD)outQueueSize);	
	
	// purse port
	PurgeComm(hComPort_, PURGE_TXABORT | PURGE_TXCLEAR | PURGE_RXABORT | PURGE_RXCLEAR);
	
	// set timeouts
	COMMTIMEOUTS timeouts;
	timeouts.ReadIntervalTimeout = 0xFFFFFFFF;
	timeouts.ReadTotalTimeoutMultiplier = 0;
	timeouts.ReadTotalTimeoutConstant = 0;
	timeouts.WriteTotalTimeoutMultiplier = 2 * CBR_9600 / baudRate;
	timeouts.WriteTotalTimeoutConstant = 0;
	SetCommTimeouts(hComPort_, &timeouts);
	
	// set dcb
	DCB dcb;
	dcb.DCBlength = sizeof(DCB);
	GetCommState(hComPort_, &dcb);
	//dcb.fBinary = TRUE;  // Windows does not support nonbinary mode transfers, so this member must be TRUE
	dcb.BaudRate = baudRate;
	dcb.ByteSize = 8;
	dcb.Parity = NOPARITY;  // no parity
	dcb.StopBits = ONESTOPBIT;  // 1 stop bit
	//dcb.fInX = dcb.fOutX = TRUE;  // use Xon & Xoff
	//dcb.XonChar = PE_SERIAL_PROTOCOL__ASCII_XON;
	//dcb.XoffChar = PE_SERIAL_PROTOCOL__ASCII_XOFF;
	//dcb.XonLim = 100;
	//dcb.XoffLim = 100;
	if (!::SetCommState(hComPort_, &dcb))
	{
		std::cerr << "SetCommState failed with error: " << GetLastError() << std::endl;
		return false;
	}

	return true;
}

void WinSerialPort::disconnect()
{
	if (INVALID_HANDLE_VALUE != hComPort_)
	{
		SetCommMask(hComPort_, 0);
		PurgeComm(hComPort_, PURGE_TXABORT | PURGE_TXCLEAR | PURGE_RXABORT | PURGE_RXCLEAR);
		CloseHandle(hComPort_);
		hComPort_ = INVALID_HANDLE_VALUE;
	}

	ResetEvent(ovSend_.hEvent);
	ResetEvent(ovRecv_.hEvent);
}

WinSerialPort::EState WinSerialPort::waitFor(OVERLAPPED& ovWait, const unsigned long timeoutInterval, const bool isSending, size_t& transferredBytes)
{
/*
	if (!::GetOverlappedResult(hComPort_, &ovWait, (DWORD *)&transferredBytes, TRUE))
	//if (!::GetOverlappedResult(hComPort_, &ovWait, (DWORD *)&transferredBytes, FALSE))
	{
		const DWORD err = GetLastError();
		std::cerr << (isSending ? "send error: " : "recv error: ") << err << std::endl;
		if (ERROR_IO_INCOMPLETE != err)
		{
			clearError();
			return WinSerialPort::E_ERROR;
		}
	}

	return WinSerialPort::E_OK;
*/
	switch(::WaitForSingleObject(ovWait.hEvent, timeoutInterval))
	{
	case WAIT_OBJECT_0:
		if (!GetOverlappedResult(hComPort_, &ovWait, (DWORD *)&transferredBytes, FALSE))
		{
			const DWORD err = GetLastError();
			std::cerr << (isSending ? "sending wait error: " : "receiving wait error: ") << err << std::endl;
			if (ERROR_IO_INCOMPLETE != err)
			{
				clearError();
				return WinSerialPort::E_ERROR;
			}
		}
		return WinSerialPort::E_OK;
	case WAIT_TIMEOUT:
		return WinSerialPort::E_TIMEOUT;
	default:
		return WinSerialPort::E_ERROR;
	}
}

WinSerialPort::EState WinSerialPort::send(const unsigned char* data, const size_t dataSize, const unsigned long timeoutInterval_msec /*= 100*/)
{
	size_t sentBytes, totalSentBytes = 0;
	while (totalSentBytes < dataSize)
	{
		sentBytes = 0;
		if (WriteFile(hComPort_, data, (DWORD)dataSize, (DWORD *)&sentBytes, &ovSend_))
		{
			if (0 == sentBytes)
			{
				// FIXME [check] >>
				std::cout << "send 0 byte" << std::endl;
				continue;
			}
		}
		else
		{
			if (ERROR_IO_PENDING == GetLastError())  // I/O operation pending
			{
				const WinSerialPort::EState state = waitFor(ovSend_, timeoutInterval_msec, true, sentBytes);
				if (WinSerialPort::E_OK != state) return state;
			}
			else
			{
				std::cerr << "wait failed with error: " << GetLastError() << std::endl;
				clearError();
				return WinSerialPort::E_ERROR;
			}
		}

		totalSentBytes += sentBytes;
	}
	
	return WinSerialPort::E_OK;
}

WinSerialPort::EState WinSerialPort::receive(GuardedByteBuffer& recvBuf, const unsigned long timeoutInterval_msec /*= 100*/, const size_t bufferLen /*= 0*/)
{
	//
	DWORD eventMask = 0L;
	if (WaitCommEvent(hComPort_, &eventMask, &ovRecv_))
	{
		if ((EV_DSR & eventMask) == EV_DSR)
		{
			std::cout << "DSR" << std::endl;
			// TODO [add] >>
		}
		if ((EV_CTS & eventMask) == EV_CTS)
		{
			std::cout << "CTS" << std::endl;
			// TODO [add] >>
		}
		if ((EV_RXCHAR & eventMask) == EV_RXCHAR)
		{
			std::cout << "RXCHAR" << std::endl;
			// TODO [add] >>
		}
	}
	else
	{
		if (ERROR_IO_PENDING == GetLastError())  // I/O operation pending
		{
			size_t recvBytes = 0;
			const WinSerialPort::EState state = waitFor(ovRecv_, timeoutInterval_msec, false, recvBytes);
			if (WinSerialPort::E_OK != state) return state;
		}
		else
		{
			std::cerr << "wait failed with error: " << GetLastError() << std::endl;
			//clearError();
			return WinSerialPort::E_ERROR;
		}
	}

	//
	size_t recvBytes = 0;
	const size_t bufLen = bufferLen ? bufferLen : 8192;

	std::vector<unsigned char> buf(bufLen, 0);
	if (::ReadFile(hComPort_, &buf[0], (DWORD)bufLen, (DWORD *)&recvBytes, &ovRecv_))
	{
		if (0 == recvBytes)
		{
			// FIXME [check] >>
			std::cout << "receive 0 byte" << std::endl;
			return WinSerialPort::E_OK;
		}
	}
	else
	{
		switch (GetLastError())
		{
		case ERROR_HANDLE_EOF:
			return WinSerialPort::E_OK;
		case ERROR_IO_PENDING:  // I/O operation pending
			{
				const WinSerialPort::EState state = waitFor(ovRecv_, timeoutInterval_msec, false, recvBytes);
				if (WinSerialPort::E_OK != state) return state;
			}
			break;
		default:
			std::cerr << "wait failed with error: " << GetLastError() << std::endl;;
			clearError();
			return WinSerialPort::E_ERROR;
		}
	}

	return recvBuf.push((GuardedByteBuffer::value_type *)(&buf[0]), (size_t)recvBytes) ? WinSerialPort::E_OK : WinSerialPort::E_ERROR;
}

bool WinSerialPort::cancelIo()
{
	return ::CancelIo(hComPort_) == TRUE;
}

void WinSerialPort::clearError()
{
	DWORD errorFlags;
	COMSTAT	comstat;
	ClearCommError(hComPort_, &errorFlags, &comstat);
}

}  // namespace swl
