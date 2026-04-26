#include <cstdio>
#include <string>
#include <vector>

#include "DTLN_AEC.h"

namespace {
constexpr int kWaveHeaderSize = 44;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "Usage: %s <input_ref.wav> <input_rec.wav> <output.pcm>\n",
                 argv[0]);
    return 1;
  }

  const std::string input_ref_wave(argv[1]);
  const std::string input_rec_wave(argv[2]);
  const std::string output_wave(argv[3]);

  FILE* input_ref_file = std::fopen(input_ref_wave.c_str(), "rb");
  FILE* input_rec_file = std::fopen(input_rec_wave.c_str(), "rb");
  FILE* output_file = std::fopen(output_wave.c_str(), "wb+");

  if (input_ref_file == nullptr || input_rec_file == nullptr ||
      output_file == nullptr) {
    std::fprintf(stderr, "Failed to open one or more files.\n");
    if (input_ref_file != nullptr) std::fclose(input_ref_file);
    if (input_rec_file != nullptr) std::fclose(input_rec_file);
    if (output_file != nullptr) std::fclose(output_file);
    return 1;
  }

  DTLN_AEC dtln_aec;
  const int frame_size = dtln_aec.Init();
  if (frame_size <= 0) {
    std::fprintf(stderr, "DTLN_AEC initialization failed.\n");
    std::fclose(input_ref_file);
    std::fclose(input_rec_file);
    std::fclose(output_file);
    return 1;
  }

  std::vector<short> input_ref_sample(frame_size);
  std::vector<short> input_rec_sample(frame_size);
  std::vector<short> output_sample(frame_size);

  // Skip wave header.
  std::fread(input_ref_sample.data(), 1, kWaveHeaderSize, input_ref_file);
  std::fread(input_rec_sample.data(), 1, kWaveHeaderSize, input_rec_file);

  while (true) {
    int read_size = std::fread(input_ref_sample.data(), 1,
                               frame_size * sizeof(short), input_ref_file);
    if (read_size <= 0) {
      break;
    }

    read_size = std::fread(input_rec_sample.data(), 1,
                           frame_size * sizeof(short), input_rec_file);
    if (read_size <= 0) {
      break;
    }

    dtln_aec.Process(input_ref_sample.data(), input_rec_sample.data(),
                     output_sample.data());

    std::fwrite(output_sample.data(), 1, frame_size * sizeof(short),
                output_file);
  }

  std::fclose(input_ref_file);
  std::fclose(input_rec_file);
  std::fclose(output_file);

  return 0;
}
