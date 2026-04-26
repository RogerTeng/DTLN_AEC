#ifndef DTLN_AEC_DTLN_AEC_H_
#define DTLN_AEC_DTLN_AEC_H_

#if defined(_WIN32) || defined(_WIN64)

#ifdef DTLNAEC_EXPORTS
#define DTLNAEC __declspec(dllexport)
#else
#define DTLNAEC __declspec(dllimport)
#endif

// Only support 16K 16Bit Mono PCM.
class DTLNAEC DTLN_AEC
#else
class DTLN_AEC
#endif
{
 public:
  DTLN_AEC();
  ~DTLN_AEC();

  // Returns number of input samples, -1 = fail.
  int Init();

  // 0 = success, -1 = fail.
  int Process(short* ref_buffer, short* rec_buffer, short* output_buffer);

 private:
  class Impl;
  Impl* impl_ = nullptr;
};

#endif  // DTLN_AEC_DTLN_AEC_H_
