

// NOLINTBEGIN
#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/c/common.h>

#include <climits>
#include <cmath>
#include <cstring>

#include "dtln_aec_128_1.h"
#include "dtln_aec_128_2.h"

// Use KissFFT https://github.com/mborgerding/kissfft
#include "DTLN_AEC.h"
#include "kiss_fftr.h"
// NOLINTEND

// 1 Network contain 2 models
// Please check : https://github.com/breizhn/DTLN-aec
// This code is translate from :
// https://github.com/breizhn/DTLN-aec/blob/main/run_aec.py

// const param
constexpr int kWindowSize = 512;
constexpr int kWindowShift = 128;
constexpr int kFftForTensorSize = (kWindowSize / 2 + 1);

constexpr int kNumModels = 2;

constexpr int kNumThreads = 1;

class DTLN_AEC::Impl {
 public:
  int Init();
  void Release();

  int Process(short *lpsRefBuffer, short *lpsRecBuffer, short *lpsOutputBuffer);
  void AEC();

  TfLiteModel *m_lppoTfliteModel[kNumModels];
  TfLiteInterpreter *m_lppoInterpreter[kNumModels];
  TfLiteInterpreterOptions *m_lpoInterpreterOptions = nullptr;

  TfLiteTensor *m_lppoInputTensor[kNumModels][3];
  const TfLiteTensor *m_lppoOutputTensor[kNumModels][2];

  // FFT
  kiss_fftr_cfg m_lpoFftrCfg = nullptr;
  kiss_fftr_cfg m_lpoIfftrCfg = nullptr;

  kiss_fft_cpx *m_lpoInputRefCpx = nullptr;
  kiss_fft_cpx *m_lpoInputRecCpx = nullptr;
  kiss_fft_cpx *m_lpoOutputCpx = nullptr;

  // Internal buffer
  float *m_lpfInputRefBuffer = nullptr;
  float *m_lpfInputRecBuffer = nullptr;
  float *m_lpfOutputBuffer = nullptr;

  float *m_lpfDtlnFreqOutput = nullptr;
  float *m_lpfDtlnTimeOutput = nullptr;

  int m_lpnStateSize[kNumModels];
  float *m_lppfStates[kNumModels];

  float *m_lpfInputRefMag = nullptr;
  float *m_lpfInputRefPhase = nullptr;

  float *m_lpfInputRecMag = nullptr;
  float *m_lpfInputRecPhase = nullptr;

  float *m_lpfEstimatedBlock = nullptr;

  // Format change buffer
  float *m_lpfInputRefSample = nullptr;
  float *m_lpfInputRecSample = nullptr;
  float *m_lpfOutputSample = nullptr;

  bool m_bInitSuccess = false;
};

int DTLN_AEC::Impl::Init() {
  int nRet = -1;

  do {
    for (int i = 0; i < kNumModels; i++) {
      m_lppoTfliteModel[i] = nullptr;
      m_lppoInterpreter[i] = nullptr;

      m_lppfStates[i] = nullptr;
    }

    // Load models
    m_lppoTfliteModel[0] =
        TfLiteModelCreate(k_lpszModel1Tflite, k_nModel1TfliteLen);
    m_lppoTfliteModel[1] =
        TfLiteModelCreate(k_lpszModel2Tflite, k_nModel2TfliteLen);

    if (m_lppoTfliteModel[0] == nullptr || m_lppoTfliteModel[1] == nullptr)
      break;

    // Create option
    m_lpoInterpreterOptions = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(m_lpoInterpreterOptions, kNumThreads);

    // Create the interpreter
    m_lppoInterpreter[0] =
        TfLiteInterpreterCreate(m_lppoTfliteModel[0], m_lpoInterpreterOptions);
    m_lppoInterpreter[1] =
        TfLiteInterpreterCreate(m_lppoTfliteModel[1], m_lpoInterpreterOptions);

    if (m_lppoInterpreter[0] == nullptr || m_lppoInterpreter[1] == nullptr)
      break;

    // Allocate tensor
    if (TfLiteInterpreterAllocateTensors(m_lppoInterpreter[0]) != kTfLiteOk)
      break;
    if (TfLiteInterpreterAllocateTensors(m_lppoInterpreter[1]) != kTfLiteOk)
      break;

    // When use original model
    // Input tensor order:
    // Model_1[] = {rec, state, ref}
    // Model_2[] = {est, state, ref}
    // When use quantized models in PiDTLN
    // Input tensor order:
    // Model_1[] = {rec, ref, state}
    // Model_2[] = {ref, state, est}
    for (int i = 0; i < kNumModels; i++) {
      m_lppoInputTensor[i][0] =
          TfLiteInterpreterGetInputTensor(m_lppoInterpreter[i], 0);
      m_lppoInputTensor[i][1] =
          TfLiteInterpreterGetInputTensor(m_lppoInterpreter[i], 1);
      m_lppoInputTensor[i][2] =
          TfLiteInterpreterGetInputTensor(m_lppoInterpreter[i], 2);

      m_lppoOutputTensor[i][0] =
          TfLiteInterpreterGetOutputTensor(m_lppoInterpreter[i], 0);
      m_lppoOutputTensor[i][1] =
          TfLiteInterpreterGetOutputTensor(m_lppoInterpreter[i], 1);

      m_lpnStateSize[i] = m_lppoInputTensor[i][1]->bytes / sizeof(float);
    }

    // RFFT/iRFFT
    m_lpoFftrCfg = kiss_fftr_alloc(kWindowSize, 0, 0, 0);
    m_lpoIfftrCfg = kiss_fftr_alloc(kWindowSize, 1, 0, 0);

    m_lpoInputRefCpx = new kiss_fft_cpx[kFftForTensorSize];
    m_lpoInputRecCpx = new kiss_fft_cpx[kFftForTensorSize];
    m_lpoOutputCpx = new kiss_fft_cpx[kFftForTensorSize];

    // Internal buffer
    m_lpfInputRefBuffer = new float[kWindowSize];
    m_lpfInputRecBuffer = new float[kWindowSize];
    m_lpfOutputBuffer = new float[kWindowSize];

    memset(m_lpfInputRefBuffer, 0, kWindowSize * sizeof(float));
    memset(m_lpfInputRecBuffer, 0, kWindowSize * sizeof(float));
    memset(m_lpfOutputBuffer, 0, kWindowSize * sizeof(float));

    m_lpfDtlnFreqOutput = new float[kFftForTensorSize];
    m_lpfDtlnTimeOutput = new float[kWindowSize];

    memset(m_lpfDtlnFreqOutput, 0, kFftForTensorSize * sizeof(float));
    memset(m_lpfDtlnTimeOutput, 0, kWindowSize * sizeof(float));

    m_lpfInputRefMag = new float[kFftForTensorSize];
    m_lpfInputRefPhase = new float[kFftForTensorSize];

    memset(m_lpfInputRefMag, 0, kFftForTensorSize * sizeof(float));
    memset(m_lpfInputRefPhase, 0, kFftForTensorSize * sizeof(float));

    m_lpfInputRecMag = new float[kFftForTensorSize];
    m_lpfInputRecPhase = new float[kFftForTensorSize];

    memset(m_lpfInputRecMag, 0, kFftForTensorSize * sizeof(float));
    memset(m_lpfInputRecPhase, 0, kFftForTensorSize * sizeof(float));

    m_lpfEstimatedBlock = new float[kWindowSize];

    memset(m_lpfEstimatedBlock, 0, kWindowSize * sizeof(float));

    for (int i = 0; i < kNumModels; i++) {
      m_lppfStates[i] = new float[m_lpnStateSize[i]];
      memset(m_lppfStates[i], 0, m_lpnStateSize[i] * sizeof(float));
    }

    // Format change buffer
    m_lpfInputRefSample = new float[kWindowSize];
    m_lpfInputRecSample = new float[kWindowSize];
    m_lpfOutputSample = new float[kWindowSize];

    memset(m_lpfInputRefSample, 0, kWindowSize * sizeof(float));
    memset(m_lpfInputRecSample, 0, kWindowSize * sizeof(float));
    memset(m_lpfOutputSample, 0, kWindowSize * sizeof(float));

    m_bInitSuccess = true;

    nRet = kWindowSize;

  } while (0);

  return nRet;
}

void DTLN_AEC::Impl::Release() {
  // Tensorflow lite
  for (int i = 0; i < kNumModels; i++) {
    if (m_lppoTfliteModel[i] != nullptr)
      TfLiteModelDelete(m_lppoTfliteModel[i]);

    if (m_lppoInterpreter[i] != nullptr)
      TfLiteInterpreterDelete(m_lppoInterpreter[i]);
  }

  if (m_lpoInterpreterOptions != nullptr)
    TfLiteInterpreterOptionsDelete(m_lpoInterpreterOptions);

  // RFFT/iRFFT
  if (m_lpoFftrCfg != nullptr) kiss_fft_free(m_lpoFftrCfg);

  if (m_lpoIfftrCfg != nullptr) kiss_fft_free(m_lpoIfftrCfg);

  if (m_lpoInputRefCpx != nullptr) delete[] m_lpoInputRefCpx;

  if (m_lpoInputRecCpx != nullptr) delete[] m_lpoInputRecCpx;

  if (m_lpoOutputCpx != nullptr) delete[] m_lpoOutputCpx;

  // Internal buffer
  if (m_lpfInputRefBuffer != nullptr) delete[] m_lpfInputRefBuffer;

  if (m_lpfInputRecBuffer != nullptr) delete[] m_lpfInputRecBuffer;

  if (m_lpfOutputBuffer != nullptr) delete[] m_lpfOutputBuffer;

  if (m_lpfDtlnFreqOutput != nullptr) delete[] m_lpfDtlnFreqOutput;

  if (m_lpfDtlnTimeOutput != nullptr) delete[] m_lpfDtlnTimeOutput;

  for (int i = 0; i < kNumModels; i++) {
    if (m_lppfStates[i] != nullptr) delete[] m_lppfStates[i];
  }

  if (m_lpfInputRefMag != nullptr) delete[] m_lpfInputRefMag;

  if (m_lpfInputRefPhase != nullptr) delete[] m_lpfInputRefPhase;

  if (m_lpfInputRecMag != nullptr) delete[] m_lpfInputRecMag;

  if (m_lpfInputRecPhase != nullptr) delete[] m_lpfInputRecPhase;

  if (m_lpfEstimatedBlock != nullptr) delete[] m_lpfEstimatedBlock;

  // Format change buffer
  if (m_lpfInputRefSample != nullptr) delete[] m_lpfInputRefSample;

  if (m_lpfInputRecSample != nullptr) delete[] m_lpfInputRecSample;

  if (m_lpfOutputSample != nullptr) delete[] m_lpfOutputSample;
}

int DTLN_AEC::Impl::Process(short *lpsRefBuffer, short *lpsRecBuffer,
                            short *lpsOutputBuffer) {
  int nRet = -1;

  do {
    if (m_bInitSuccess == false) break;

    if (lpsRefBuffer == nullptr || lpsRecBuffer == nullptr ||
        lpsOutputBuffer == nullptr)
      break;

    // Convert short to float
    for (int i = 0; i < kWindowSize; i++) {
      m_lpfInputRefSample[i] = (float)lpsRefBuffer[i] * 1.0f / SHRT_MAX;
    }

    for (int i = 0; i < kWindowSize; i++) {
      m_lpfInputRecSample[i] = (float)lpsRecBuffer[i] * 1.0f / SHRT_MAX;
    }

    AEC();

    // Convert float to short
    for (int i = 0; i < kWindowSize; i++) {
      lpsOutputBuffer[i] = (short)(m_lpfOutputSample[i] * SHRT_MAX);
    }

    nRet = 0;
  } while (0);

  return nRet;
}

void DTLN_AEC::Impl::AEC() {
  int nNumBlocks = kWindowSize / kWindowShift;

  float *pfInputRefSample = m_lpfInputRefSample;
  float *pfInputRecSample = m_lpfInputRecSample;
  float *pfOutputSample = m_lpfOutputSample;

  for (int i = 0; i < nNumBlocks; i++) {
    // Buffer shift to match FFT size
    memmove(m_lpfInputRefBuffer, m_lpfInputRefBuffer + kWindowShift,
            (kWindowSize - kWindowShift) * sizeof(float));
    memcpy(m_lpfInputRefBuffer + (kWindowSize - kWindowShift), pfInputRefSample,
           kWindowShift * sizeof(float));

    memmove(m_lpfInputRecBuffer, m_lpfInputRecBuffer + kWindowShift,
            (kWindowSize - kWindowShift) * sizeof(float));
    memcpy(m_lpfInputRecBuffer + (kWindowSize - kWindowShift), pfInputRecSample,
           kWindowShift * sizeof(float));

    // Prepare buffer
    memset(m_lpfInputRefMag, 0, kFftForTensorSize * sizeof(float));
    memset(m_lpfInputRefPhase, 0, kFftForTensorSize * sizeof(float));

    memset(m_lpfInputRecMag, 0, kFftForTensorSize * sizeof(float));
    memset(m_lpfInputRecPhase, 0, kFftForTensorSize * sizeof(float));

    memset(m_lpfEstimatedBlock, 0, kWindowSize * sizeof(float));

    // Use RFFT/iRFFT to implement STFT/iSTFT

    // RFFT
    kiss_fftr(m_lpoFftrCfg, m_lpfInputRefBuffer, m_lpoInputRefCpx);
    kiss_fftr(m_lpoFftrCfg, m_lpfInputRecBuffer, m_lpoInputRecCpx);

    // Calculate Mag/Phase
    for (int j = 0; j < kFftForTensorSize; j++) {
      // How to calculate Mag/Phase:
      // check 3a/3b in
      // https://www.gaussianwaves.com/2015/11/interpreting-fft-results-obtaining-magnitude-and-phase-information/
      m_lpfInputRefMag[j] =
          sqrtf(m_lpoInputRefCpx[j].r * m_lpoInputRefCpx[j].r +
                m_lpoInputRefCpx[j].i * m_lpoInputRefCpx[j].i);
      m_lpfInputRefPhase[j] =
          atan2f(m_lpoInputRefCpx[j].i, m_lpoInputRefCpx[j].r);

      m_lpfInputRecMag[j] =
          sqrtf(m_lpoInputRecCpx[j].r * m_lpoInputRecCpx[j].r +
                m_lpoInputRecCpx[j].i * m_lpoInputRecCpx[j].i);
      m_lpfInputRecPhase[j] =
          atan2f(m_lpoInputRecCpx[j].i, m_lpoInputRecCpx[j].r);
    }

    // Set data into tensor
    TfLiteTensorCopyFromBuffer(m_lppoInputTensor[0][0], m_lpfInputRecMag,
                               kFftForTensorSize * sizeof(float));
    TfLiteTensorCopyFromBuffer(m_lppoInputTensor[0][1], m_lppfStates[0],
                               m_lpnStateSize[0] * sizeof(float));
    TfLiteTensorCopyFromBuffer(m_lppoInputTensor[0][2], m_lpfInputRefMag,
                               kFftForTensorSize * sizeof(float));

    // DTLN for freq domain
    TfLiteInterpreterInvoke(m_lppoInterpreter[0]);

    // Get data from tensor
    TfLiteTensorCopyToBuffer(m_lppoOutputTensor[0][0], m_lpfDtlnFreqOutput,
                             kFftForTensorSize * sizeof(float));
    TfLiteTensorCopyToBuffer(m_lppoOutputTensor[0][1], m_lppfStates[0],
                             m_lpnStateSize[0] * sizeof(float));

    // iRFFT
    // m_lpfDtlnFreqOutput is out_mask
    // Use orignal Mag/Phase to restore generated freq
    for (int j = 0; j < kFftForTensorSize; j++) {
      // Re{ z } = Re{ a + ib } = Mag * cos[φ] * freq
      // Im{ z } = Im{ a + ib } = Mag * sin[φ] * freq
      m_lpoOutputCpx[j].r = m_lpfInputRecMag[j] * cosf(m_lpfInputRecPhase[j]) *
                            m_lpfDtlnFreqOutput[j];
      m_lpoOutputCpx[j].i = m_lpfInputRecMag[j] * sinf(m_lpfInputRecPhase[j]) *
                            m_lpfDtlnFreqOutput[j];
    }

    kiss_fftri(m_lpoIfftrCfg, m_lpoOutputCpx, m_lpfEstimatedBlock);

    // FFT coefficient 1/N
    for (int j = 0; j < kWindowSize; j++)
      m_lpfEstimatedBlock[j] = m_lpfEstimatedBlock[j] / kWindowSize;

    // Set data into tensor
    TfLiteTensorCopyFromBuffer(m_lppoInputTensor[1][0], m_lpfEstimatedBlock,
                               kWindowSize * sizeof(float));
    TfLiteTensorCopyFromBuffer(m_lppoInputTensor[1][1], m_lppfStates[1],
                               m_lpnStateSize[1] * sizeof(float));
    TfLiteTensorCopyFromBuffer(m_lppoInputTensor[1][2], m_lpfInputRefBuffer,
                               kWindowSize * sizeof(float));

    // DTLN for time domain
    TfLiteInterpreterInvoke(m_lppoInterpreter[1]);

    // Get data from tensor
    TfLiteTensorCopyToBuffer(m_lppoOutputTensor[1][0], m_lpfDtlnTimeOutput,
                             kWindowSize * sizeof(float));
    TfLiteTensorCopyToBuffer(m_lppoOutputTensor[1][1], m_lppfStates[1],
                             m_lpnStateSize[1] * sizeof(float));

    // Overlap add
    memmove(m_lpfOutputBuffer, m_lpfOutputBuffer + kWindowShift,
            (kWindowSize - kWindowShift) * sizeof(float));
    memset(m_lpfOutputBuffer + (kWindowSize - kWindowShift), 0,
           kWindowShift * sizeof(float));

    for (int j = 0; j < kWindowSize; j++)
      m_lpfOutputBuffer[j] += m_lpfDtlnTimeOutput[j];

    memcpy(pfOutputSample, m_lpfOutputBuffer, kWindowShift * sizeof(float));

    pfInputRefSample += kWindowShift;
    pfInputRecSample += kWindowShift;
    pfOutputSample += kWindowShift;
  }
}

DTLN_AEC::DTLN_AEC() : impl_(new DTLN_AEC::Impl) {}

DTLN_AEC::~DTLN_AEC() {
  impl_->Release();

  delete impl_;
  impl_ = nullptr;
}

int DTLN_AEC::Init() { return impl_->Init(); }

int DTLN_AEC::Process(short *lpsRefBuffer, short *lpsRecBuffer,
                      short *lpsOutputBuffer) {
  return impl_->Process(lpsRefBuffer, lpsRecBuffer, lpsOutputBuffer);
}
