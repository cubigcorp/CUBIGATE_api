# CUBIGate

## Directory Structure

    .
    ├── CUBIGate               # CUBIGate base directory
    ├── data                   # Synthetic data directory for reports
    └── dp                     # DP library `DPSDA`
          ├── apis             # APIs to generate samples
          ├── dpsda            # DP main library
          ├── data             # Real data directory to learn distribution
          └── script           # Script files to run `DPSDA``
    └── scripts                # Script files to test synthetic data and make reports


## Files

* `classify.py`: Fine-tune a pretrained classifier
  
  ```
  classify.py --data_dir [DATA_DIR] --prefix [PREFIX] --device [DEVICE_NUM] --num_classes [NUM_CLASSES] --train --valid --test
  ```
  * DATA_DIR: Path of the base data directory, may contain sub directories for train/test/valid datasets.
  * PREFIX: Prefix for output files.
  * DEVICE_NUM: Index of the GPU device to put everything on.
  * NUM_CLASSES: Number of classes.
  * train, valid, test: Optional, whether to train, validate, test the model.

* `ClassifyDataset.py`: Dataset for classification
  
* `clip_classify.py`: CLIP zero-shot classification
  
  ```
  clip_classify.py --data_dir [DATA_DIR] --dataset [DATASET_NAME] --device [DEVICE_NUM] --num_classes [NUM_CLASSES] --result_dir [RESULT_DIR] --dp
  ```
  * DATA_DIR: Path of the data directory.
  * DATASET_NAME: Name of the dataset to classify.
  * DEVICE: Index of the GPU device to put everything on.
  * NUM_CLASSES: Number of classes.
  * RESULT_DIR: Path of the directory where the result will be saved.
  * dp: Optional, Whether the dataset is dp or non-dp.
  
* `cubig_gen_infer.py`: Generate synthetic data out of learned distribution

* `lora.py`: Train LoRA

* `LoraDataset.py`: Dataset for LoRA

* `preprocess.py`: Process datasets for DPSDA and classification