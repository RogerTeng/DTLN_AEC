

#include <string>
#include "DTLN_AEC.h"

int main(int argc, char *argv[])
{
    //lpszInputRefWave is a reference data(far end)
    //lpszInputRecWave is a recording data(near end recording)

    std::string lpszInputRefWave = std::string(argv[1]);
    std::string lpszInputRecWave = std::string(argv[2]);
    std::string lpszOutputWave = std::string(argv[3]);

    FILE *lpoInputRefFile = NULL;
    FILE *lpoInputRecFile = NULL;
    FILE *lpoOutputFile = NULL;

    short *lpsInputRefSample = NULL;
    short *lpsInputRecSample = NULL;
    short *lpsOutputSample = NULL;

    int nReadSize, nFrameSize;

    lpoInputRefFile = fopen(lpszInputRefWave.c_str(), "rb");
    lpoInputRecFile = fopen(lpszInputRecWave.c_str(), "rb");
    lpoOutputFile = fopen(lpszOutputWave.c_str(), "wb+");

    DTLN_AEC oDtlnAec;

    nFrameSize = oDtlnAec.Init();

    lpsInputRefSample = new short[nFrameSize];
    lpsInputRecSample = new short[nFrameSize];
    lpsOutputSample = new short[nFrameSize];

    //Skip wave header
    fread(lpsInputRefSample, 1, 44, lpoInputRefFile);
    fread(lpsInputRecSample, 1, 44, lpoInputRecFile);

    while (true)
    {
        nReadSize = fread(lpsInputRefSample, 1, nFrameSize * sizeof(short), lpoInputRefFile);
        if (nReadSize <= 0)
            break;

        nReadSize = fread(lpsInputRecSample, 1, nFrameSize * sizeof(short), lpoInputRecFile);
        if (nReadSize <= 0)
            break;

        oDtlnAec.Process(lpsInputRefSample, lpsInputRecSample, lpsOutputSample);

        //write PCM
        fwrite(lpsOutputSample, 1, nFrameSize * sizeof(short), lpoOutputFile);
    }

    fclose(lpoInputRefFile);
    fclose(lpoInputRecFile);
    fclose(lpoOutputFile);

    if (lpsInputRefSample != NULL)
        delete[] lpsInputRefSample;

    if (lpsInputRecSample != NULL)
        delete[] lpsInputRecSample;

    if (lpsOutputSample != NULL)
        delete[] lpsOutputSample;

    return 0;

}