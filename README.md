# DTLN_AEC
A stand alone wrapper library in C code of [DTLN-aec](https://github.com/breizhn/DTLN-aec).<br/>
Inlcude prebuilt TensorFlow Lite v2.5.2 for Windows x64/macOS(Intel)/macOS(Apple Silicon).

If you need NR, please check [DTLN_NR](https://github.com/RogerTeng/DTLN_NR).

#### This repository provide VS2019 project, but you could build this source code in xCode.

I use hardcode model(dtln_aec_128) in this project, if you want change model,<br/>
Use
	
	TfLiteModelCreateFromFile()


To replace
	
	TfLiteModelCreate()
## Note
If you want use quantized models in [PiDTLN](https://github.com/SaneBow/PiDTLN),</br>
MUST check tensor input,</br>
PiDTLN use diffrent tensor input order with original DTLN-aec.

## Acknowledgement
* This project is based on the [DTLN-aec](https://github.com/breizhn/DTLN-aec) by [breizhn](https://github.com/breizhn).
* FFT from [KISS FFT](https://github.com/mborgerding/kissfft) by [mborgerding](https://github.com/mborgerding).

## Citing

```BibTex
@INPROCEEDINGS{westhausen21_dtln_aec,
  author={Westhausen, Nils L. and Meyer, Bernd T.},
  booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={{Acoustic Echo Cancellation with the Dual-Signal Transformation LSTM Network}}, 
  year={2021},
  volume={},
  number={},
  pages={7138-7142},
  doi={10.1109/ICASSP39728.2021.9413510}
  }
