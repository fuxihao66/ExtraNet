# ExtraNet
This is the repository for the tools and network source codes for our paper [ExtraNet: Real-time Extrapolated Rendering for Low-latency Temporal Supersampling][ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia 2021)].

## How to generate dataset
We provide a custom built Unreal Engine for generating dataset. Any Unreal Engine scenes requiring engine version <= 4.25.3 can be opened by our build.
The link of our build: https://drive.google.com/drive/folders/1ZUbct6Z2T3gGhbxjQU0-u2PPflwYv-UZ?usp=sharing

Make sure to enable DX12 RHI and Ray tracing in Unreal settings.

This custom build is modified by Hengjun Ma.
### Generate raw data
We use Unreal Engine's sequence to generate data. More information about sequence and Sequencer, please refer to Unreal Engine documentation.
For a recorded sequence, settings shown below are required before capturing frames:

![ue](ue.png)


Make sure to store frames from different sequences in different folders.  

### Preprocessing
Raw data needs to be preprocessed before doing training or testing. We provide a script "preprocess.py" to generate training/testing dataset from raw data.
The usage of script is:
```python
python preprocess.py \some_path\raw_data_folder
```
For better efficiency of data loading, we also provide a script compressData.py for compressing data into .npz file. 
Usage: comming soon...


Currently our script has only been tested on Windows and accepts only "\\" in file path. In addition, there are some settings (like thread number) can only be editted in script. We might commit some modification later to provide better user experience.

## Model
### Training
The training script is: Model/train.py.
```python
python train.py
```
### Testing
The testing script is: Model/inference.py.
```python
python inference.py
```
## Integration into Unreal Engine
https://github.com/fuxihao66/UnrealEngine/tree/UE425ExtraNet_active (still under development..)


## Citation
If you find the tools or codes useful in your research, please cite:
```
@article{10.1145/3478513.3480531,
author = {Guo, Jie and Fu, Xihao and Lin, Liqiang and Ma, Hengjun and Guo, Yanwen and Liu, Shiqiu and Yan, Ling-Qi},
title = {ExtraNet: Real-Time Extrapolated Rendering for Low-Latency Temporal Supersampling},
year = {2021},
issue_date = {December 2021},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {40},
number = {6},
doi = {10.1145/3478513.3480531},
journal = {ACM Trans. Graph.},
month = {dec},
articleno = {278},
numpages = {16}
}
```



