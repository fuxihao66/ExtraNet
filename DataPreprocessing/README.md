sky box





Raw data needs to be preprocessed before doing training or testing. We provide a script "preprocess.py" to generate training/testing dataset from raw data.
The usage of script is:
```python
python preprocess.py \some_path\raw_data_folder
```
For better efficiency of data loading, we also provide a script compressData.py for compressing data into .npz file. 
Usage: comming soon...


Currently our script has only been tested on Windows and accepts only "\\" in file path. In addition, there are some settings (like thread number) can only be editted in script. We might commit some modification later to provide better user experience.