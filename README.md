# spatial-temporal-attack
## Dataset
We collect a single-label dataset and  merge instances from different websites at different overlapping rates.

overlapping dataset: https://drive.google.com/drive/folders/1J95UaR3_b_hEg1HX3OKS9Nc22Vg5zWYo?usp=sharing

There are two datasets: synthetic and real-world. 

In the "synthetic" folder, there are train and test datasets for a single tab. We provide the Python code for mergeing single-tab dataset. You can use this to construct your own dataset. 

In the "real-world" folder, there are six 2-tab datasets. They are collected  in the real-world. The names of ".npy" files are based on their tab-opening time gaps, ranging from 0s to 60s. Moreover, we also provide a cross-protocol dataset, called "firefox.npy". This dataset is collected from the HTTPs protocol and consists of the same websites as the Tor dataset. 

The real-world dataset is collected from 15 hosts located on different regions. We totally collect more than 110,000 traces in one month and finally retain 90,000 valid traces (without Cloudflare banner and timeouts). Moreover, to select the valid traces, we train a Resnet-based website filter, filtering the invalid trace base on screenshots of webistes.

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
