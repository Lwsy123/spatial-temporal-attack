# spatial-temporal-attack
## Dataset
We collect a single-label dataset and  merge instances from different websites at different overlapping rates.

overlapping dataset: https://drive.google.com/drive/folders/1BRK21au25DiLJCjna0KfiOfzmQMZsmiL?usp=sharing

## Model
The "maincode" directory contains the prototype of our model.

~~~
maincode/
│
├── Util/ 
│   ├── Dataloader.py        # The "dataloader.py" is used to generate the train and test dataloaders
│   └── utils.py             # The "utils.py" contains the functions that are used by our model, such as the function that 
│                              generates Q, K, and V used in multi-head attention
│
├── net/
│   ├── TransformerDecoder/ # The implementation of the feature integration and correlation denoising
│   │   ├── Decoder.py 
│   │   └── DecoderBlock.py
│   ├── BertExtract.py      # The main framework of our model
│   └── BertFeatures.py     # The implementation of the feature extractor
├── main.py                 # the main code of our model
├── train.py                # model training setting
├── evaluate.py             # model evaluating setting
└── evaluation.py           # the metrics of the performance of our model

~~~
