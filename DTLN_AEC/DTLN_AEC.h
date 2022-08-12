
#if defined(_WIN32) || defined(_WIN64)

#ifdef DTLNAEC_EXPORTS
#    define DTLNAEC __declspec(dllexport)
#else
#    define DTLNAEC __declspec(dllimport)
#endif

//Only support 16K 16Bit Mono PCM

class DTLNAEC DTLN_AEC
//Windows win32/x86_64
#else
class DTLN_AEC //#elif defined(__APPLE__)
//macOS
#endif
{
public:
	DTLN_AEC();
	~DTLN_AEC();

	//Return number of input samples, -1 = Fail
	int Init(void);

	//0 = Success, -1 = Fail
	int Process(short *lpsRefBuffer, short *lpsRecBuffer, short *lpsOutputBuffer);

private:
	class m_Impl;
	m_Impl *m_lpoImpl = nullptr;
};


