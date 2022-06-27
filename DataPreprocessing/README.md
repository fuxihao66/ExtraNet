# How to create data sets from raw data

## Training sets

### Step 1: Generate all the images needed
```
python preprocess.py \some_path\raw_data_folder
```
Currently our script has only been tested on Windows and accepts only "\\" in file path. The **Thread_NUM** parameter in the script can be adjusted according to the CPU specs.

### Step 2: Compress images to .npz files
For better efficiency of data loading, we also provide a script **compressData.py** for compressing data into .npz file. 
Before running this script, the settings at the beginning should be modified:
1. Set **threadNum**
2. Set **compressedOutputDir** (This path would be the output directory of compressed data.)
3. Set **dirList** (This list represents all the folders of data sets.)
4. Set **ScenePrefix** (This setting could have been extracted from the images, and it will be fixed later.)

## Test sets
For test sets, only **preprocess.py** is needed.

(When doing inference, the skybox needs to be combined with inference output to get the final extrapolated frame (like the "Skybox.exr" file in the ExtraNet/TestData folder. For simplicity, this buffer will not be generated when exporting data, and the skybox can be extracted from "PreTonemapHDRColor.exr" by counting only those pixels with (-1,-1,-1) normals.)

## About demodulation
The **preprocess_glossiness.py** script is an alternative option for generating images. The only difference between **preprocess.py** and **preprocess_glossiness.py** is the demodulation algorithm.

In **preprocess.py**:
$$
\text{img}_{\text{demodu}} = \text{img}_{\text{pretonemap}} / \text{img}_{\text{basecolor}}.
$$

In **preprocess_glossiness.py**:
$$
\text{img}_{\text{demodu}} = \text{img}_{\text{pretonemap}} / (\text{img}_{\text{basecolor}} + \text{img}_{\text{sepcular}} * 0.08 * ( 1 - \text{img}_{\text{metallic}} )).
$$



This script works better when glossy materials take a huge part in the scene (and when BaseColor is totally dark). The scene **Bunker** is operated with this script. When using this script, the buffer **Specular** should be added when exporting data from UE.

For better demodulation methods, please refer to https://github.com/fuxihao66/Demodulation_BRDF_Pre-integration.

