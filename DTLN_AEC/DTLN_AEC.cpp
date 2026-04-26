

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

  int Process(short *ref_buffer, short *rec_buffer, short *output_buffer);
  void AEC();

  TfLiteModel *tflite_models_[kNumModels];
  TfLiteInterpreter *interpreters_[kNumModels];
  TfLiteInterpreterOptions *interpreter_options_ = nullptr;

  TfLiteTensor *input_tensors_[kNumModels][3];
  const TfLiteTensor *output_tensors_[kNumModels][2];

  // FFT
  kiss_fftr_cfg fftr_cfg_ = nullptr;
  kiss_fftr_cfg ifftr_cfg_ = nullptr;

  kiss_fft_cpx *input_ref_cpx_ = nullptr;
  kiss_fft_cpx *input_rec_cpx_ = nullptr;
  kiss_fft_cpx *output_cpx_ = nullptr;

  // Internal buffer
  float *input_ref_buffer_ = nullptr;
  float *input_rec_buffer_ = nullptr;
  float *output_buffer_ = nullptr;

  float *dtln_freq_output_ = nullptr;
  float *dtln_time_output_ = nullptr;

  int state_size_[kNumModels];
  float *states_[kNumModels];

  float *input_ref_mag_ = nullptr;
  float *input_ref_phase_ = nullptr;

  float *input_rec_mag_ = nullptr;
  float *input_rec_phase_ = nullptr;

  float *estimated_block_ = nullptr;

  // Format change buffer
  float *input_ref_sample_ = nullptr;
  float *input_rec_sample_ = nullptr;
  float *output_sample_ = nullptr;

  bool init_success_ = false;
};

int DTLN_AEC::Impl::Init() {
  int ret = -1;

  do {
    for (int i = 0; i < kNumModels; i++) {
      tflite_models_[i] = nullptr;
      interpreters_[i] = nullptr;

      states_[i] = nullptr;
    }

    // Load models
    tflite_models_[0] =
        TfLiteModelCreate(k_lpszModel1Tflite, k_nModel1TfliteLen);
    tflite_models_[1] =
        TfLiteModelCreate(k_lpszModel2Tflite, k_nModel2TfliteLen);

    if (tflite_models_[0] == nullptr || tflite_models_[1] == nullptr) break;

    // Create option
    interpreter_options_ = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(interpreter_options_, kNumThreads);

    // Create the interpreter
    interpreters_[0] =
        TfLiteInterpreterCreate(tflite_models_[0], interpreter_options_);
    interpreters_[1] =
        TfLiteInterpreterCreate(tflite_models_[1], interpreter_options_);

    if (interpreters_[0] == nullptr || interpreters_[1] == nullptr) break;

    // Allocate tensor
    if (TfLiteInterpreterAllocateTensors(interpreters_[0]) != kTfLiteOk) break;
    if (TfLiteInterpreterAllocateTensors(interpreters_[1]) != kTfLiteOk) break;

    // When use original model
    // Input tensor order:
    // Model_1[] = {rec, state, ref}
    // Model_2[] = {est, state, ref}
    // When use quantized models in PiDTLN
    // Input tensor order:
    // Model_1[] = {rec, ref, state}
    // Model_2[] = {ref, state, est}
    for (int i = 0; i < kNumModels; i++) {
      input_tensors_[i][0] =
          TfLiteInterpreterGetInputTensor(interpreters_[i], 0);
      input_tensors_[i][1] =
          TfLiteInterpreterGetInputTensor(interpreters_[i], 1);
      input_tensors_[i][2] =
          TfLiteInterpreterGetInputTensor(interpreters_[i], 2);

      output_tensors_[i][0] =
          TfLiteInterpreterGetOutputTensor(interpreters_[i], 0);
      output_tensors_[i][1] =
          TfLiteInterpreterGetOutputTensor(interpreters_[i], 1);

      state_size_[i] = input_tensors_[i][1]->bytes / sizeof(float);
    }

    // RFFT/iRFFT
    fftr_cfg_ = kiss_fftr_alloc(kWindowSize, 0, 0, 0);
    ifftr_cfg_ = kiss_fftr_alloc(kWindowSize, 1, 0, 0);

    input_ref_cpx_ = new kiss_fft_cpx[kFftForTensorSize];
    input_rec_cpx_ = new kiss_fft_cpx[kFftForTensorSize];
    output_cpx_ = new kiss_fft_cpx[kFftForTensorSize];

    // Internal buffer
    input_ref_buffer_ = new float[kWindowSize];
    input_rec_buffer_ = new float[kWindowSize];
    output_buffer_ = new float[kWindowSize];

    memset(input_ref_buffer_, 0, kWindowSize * sizeof(float));
    memset(input_rec_buffer_, 0, kWindowSize * sizeof(float));
    memset(output_buffer_, 0, kWindowSize * sizeof(float));

    dtln_freq_output_ = new float[kFftForTensorSize];
    dtln_time_output_ = new float[kWindowSize];

    memset(dtln_freq_output_, 0, kFftForTensorSize * sizeof(float));
    memset(dtln_time_output_, 0, kWindowSize * sizeof(float));

    input_ref_mag_ = new float[kFftForTensorSize];
    input_ref_phase_ = new float[kFftForTensorSize];

    memset(input_ref_mag_, 0, kFftForTensorSize * sizeof(float));
    memset(input_ref_phase_, 0, kFftForTensorSize * sizeof(float));

    input_rec_mag_ = new float[kFftForTensorSize];
    input_rec_phase_ = new float[kFftForTensorSize];

    memset(input_rec_mag_, 0, kFftForTensorSize * sizeof(float));
    memset(input_rec_phase_, 0, kFftForTensorSize * sizeof(float));

    estimated_block_ = new float[kWindowSize];

    memset(estimated_block_, 0, kWindowSize * sizeof(float));

    for (int i = 0; i < kNumModels; i++) {
      states_[i] = new float[state_size_[i]];
      memset(states_[i], 0, state_size_[i] * sizeof(float));
    }

    // Format change buffer
    input_ref_sample_ = new float[kWindowSize];
    input_rec_sample_ = new float[kWindowSize];
    output_sample_ = new float[kWindowSize];

    memset(input_ref_sample_, 0, kWindowSize * sizeof(float));
    memset(input_rec_sample_, 0, kWindowSize * sizeof(float));
    memset(output_sample_, 0, kWindowSize * sizeof(float));

    init_success_ = true;

    ret = kWindowSize;

  } while (0);

  return ret;
}

void DTLN_AEC::Impl::Release() {
  // Tensorflow lite
  for (int i = 0; i < kNumModels; i++) {
    if (tflite_models_[i] != nullptr) TfLiteModelDelete(tflite_models_[i]);

    if (interpreters_[i] != nullptr) TfLiteInterpreterDelete(interpreters_[i]);
  }

  if (interpreter_options_ != nullptr)
    TfLiteInterpreterOptionsDelete(interpreter_options_);

  // RFFT/iRFFT
  if (fftr_cfg_ != nullptr) kiss_fft_free(fftr_cfg_);

  if (ifftr_cfg_ != nullptr) kiss_fft_free(ifftr_cfg_);

  if (input_ref_cpx_ != nullptr) delete[] input_ref_cpx_;

  if (input_rec_cpx_ != nullptr) delete[] input_rec_cpx_;

  if (output_cpx_ != nullptr) delete[] output_cpx_;

  // Internal buffer
  if (input_ref_buffer_ != nullptr) delete[] input_ref_buffer_;

  if (input_rec_buffer_ != nullptr) delete[] input_rec_buffer_;

  if (output_buffer_ != nullptr) delete[] output_buffer_;

  if (dtln_freq_output_ != nullptr) delete[] dtln_freq_output_;

  if (dtln_time_output_ != nullptr) delete[] dtln_time_output_;

  for (int i = 0; i < kNumModels; i++) {
    if (states_[i] != nullptr) delete[] states_[i];
  }

  if (input_ref_mag_ != nullptr) delete[] input_ref_mag_;

  if (input_ref_phase_ != nullptr) delete[] input_ref_phase_;

  if (input_rec_mag_ != nullptr) delete[] input_rec_mag_;

  if (input_rec_phase_ != nullptr) delete[] input_rec_phase_;

  if (estimated_block_ != nullptr) delete[] estimated_block_;

  // Format change buffer
  if (input_ref_sample_ != nullptr) delete[] input_ref_sample_;

  if (input_rec_sample_ != nullptr) delete[] input_rec_sample_;

  if (output_sample_ != nullptr) delete[] output_sample_;
}

int DTLN_AEC::Impl::Process(short *ref_buffer, short *rec_buffer,
                            short *output_buffer) {
  int ret = -1;

  do {
    if (init_success_ == false) break;

    if (ref_buffer == nullptr || rec_buffer == nullptr ||
        output_buffer == nullptr)
      break;

    // Convert short to float
    for (int i = 0; i < kWindowSize; i++) {
      input_ref_sample_[i] = (float)ref_buffer[i] * 1.0f / SHRT_MAX;
    }

    for (int i = 0; i < kWindowSize; i++) {
      input_rec_sample_[i] = (float)rec_buffer[i] * 1.0f / SHRT_MAX;
    }

    AEC();

    // Convert float to short
    for (int i = 0; i < kWindowSize; i++) {
      output_buffer[i] = (short)(output_sample_[i] * SHRT_MAX);
    }

    ret = 0;
  } while (0);

  return ret;
}

void DTLN_AEC::Impl::AEC() {
  int num_blocks = kWindowSize / kWindowShift;

  float *input_ref_sample = input_ref_sample_;
  float *input_rec_sample = input_rec_sample_;
  float *output_sample = output_sample_;

  for (int i = 0; i < num_blocks; i++) {
    // Buffer shift to match FFT size
    memmove(input_ref_buffer_, input_ref_buffer_ + kWindowShift,
            (kWindowSize - kWindowShift) * sizeof(float));
    memcpy(input_ref_buffer_ + (kWindowSize - kWindowShift), input_ref_sample,
           kWindowShift * sizeof(float));

    memmove(input_rec_buffer_, input_rec_buffer_ + kWindowShift,
            (kWindowSize - kWindowShift) * sizeof(float));
    memcpy(input_rec_buffer_ + (kWindowSize - kWindowShift), input_rec_sample,
           kWindowShift * sizeof(float));

    // Prepare buffer
    memset(input_ref_mag_, 0, kFftForTensorSize * sizeof(float));
    memset(input_ref_phase_, 0, kFftForTensorSize * sizeof(float));

    memset(input_rec_mag_, 0, kFftForTensorSize * sizeof(float));
    memset(input_rec_phase_, 0, kFftForTensorSize * sizeof(float));

    memset(estimated_block_, 0, kWindowSize * sizeof(float));

    // Use RFFT/iRFFT to implement STFT/iSTFT

    // RFFT
    kiss_fftr(fftr_cfg_, input_ref_buffer_, input_ref_cpx_);
    kiss_fftr(fftr_cfg_, input_rec_buffer_, input_rec_cpx_);

    // Calculate Mag/Phase
    for (int j = 0; j < kFftForTensorSize; j++) {
      // How to calculate Mag/Phase:
      // check 3a/3b in
      // https://www.gaussianwaves.com/2015/11/interpreting-fft-results-obtaining-magnitude-and-phase-information/
      input_ref_mag_[j] = sqrtf(input_ref_cpx_[j].r * input_ref_cpx_[j].r +
                                input_ref_cpx_[j].i * input_ref_cpx_[j].i);
      input_ref_phase_[j] = atan2f(input_ref_cpx_[j].i, input_ref_cpx_[j].r);

      input_rec_mag_[j] = sqrtf(input_rec_cpx_[j].r * input_rec_cpx_[j].r +
                                input_rec_cpx_[j].i * input_rec_cpx_[j].i);
      input_rec_phase_[j] = atan2f(input_rec_cpx_[j].i, input_rec_cpx_[j].r);
    }

    // Set data into tensor
    TfLiteTensorCopyFromBuffer(input_tensors_[0][0], input_rec_mag_,
                               kFftForTensorSize * sizeof(float));
    TfLiteTensorCopyFromBuffer(input_tensors_[0][1], states_[0],
                               state_size_[0] * sizeof(float));
    TfLiteTensorCopyFromBuffer(input_tensors_[0][2], input_ref_mag_,
                               kFftForTensorSize * sizeof(float));

    // DTLN for freq domain
    TfLiteInterpreterInvoke(interpreters_[0]);

    // Get data from tensor
    TfLiteTensorCopyToBuffer(output_tensors_[0][0], dtln_freq_output_,
                             kFftForTensorSize * sizeof(float));
    TfLiteTensorCopyToBuffer(output_tensors_[0][1], states_[0],
                             state_size_[0] * sizeof(float));

    // iRFFT
    // dtln_freq_output_ is out_mask
    // Use orignal Mag/Phase to restore generated freq
    for (int j = 0; j < kFftForTensorSize; j++) {
      // Re{ z } = Re{ a + ib } = Mag * cos[φ] * freq
      // Im{ z } = Im{ a + ib } = Mag * sin[φ] * freq
      output_cpx_[j].r =
          input_rec_mag_[j] * cosf(input_rec_phase_[j]) * dtln_freq_output_[j];
      output_cpx_[j].i =
          input_rec_mag_[j] * sinf(input_rec_phase_[j]) * dtln_freq_output_[j];
    }

    kiss_fftri(ifftr_cfg_, output_cpx_, estimated_block_);

    // FFT coefficient 1/N
    for (int j = 0; j < kWindowSize; j++)
      estimated_block_[j] = estimated_block_[j] / kWindowSize;

    // Set data into tensor
    TfLiteTensorCopyFromBuffer(input_tensors_[1][0], estimated_block_,
                               kWindowSize * sizeof(float));
    TfLiteTensorCopyFromBuffer(input_tensors_[1][1], states_[1],
                               state_size_[1] * sizeof(float));
    TfLiteTensorCopyFromBuffer(input_tensors_[1][2], input_ref_buffer_,
                               kWindowSize * sizeof(float));

    // DTLN for time domain
    TfLiteInterpreterInvoke(interpreters_[1]);

    // Get data from tensor
    TfLiteTensorCopyToBuffer(output_tensors_[1][0], dtln_time_output_,
                             kWindowSize * sizeof(float));
    TfLiteTensorCopyToBuffer(output_tensors_[1][1], states_[1],
                             state_size_[1] * sizeof(float));

    // Overlap add
    memmove(output_buffer_, output_buffer_ + kWindowShift,
            (kWindowSize - kWindowShift) * sizeof(float));
    memset(output_buffer_ + (kWindowSize - kWindowShift), 0,
           kWindowShift * sizeof(float));

    for (int j = 0; j < kWindowSize; j++)
      output_buffer_[j] += dtln_time_output_[j];

    memcpy(output_sample, output_buffer_, kWindowShift * sizeof(float));

    input_ref_sample += kWindowShift;
    input_rec_sample += kWindowShift;
    output_sample += kWindowShift;
  }
}

DTLN_AEC::DTLN_AEC() : impl_(new DTLN_AEC::Impl) {}

DTLN_AEC::~DTLN_AEC() {
  impl_->Release();

  delete impl_;
  impl_ = nullptr;
}

int DTLN_AEC::Init() { return impl_->Init(); }

int DTLN_AEC::Process(short *ref_buffer, short *rec_buffer,
                      short *output_buffer) {
  return impl_->Process(ref_buffer, rec_buffer, output_buffer);
}
