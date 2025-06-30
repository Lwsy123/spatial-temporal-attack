# spatial-temporal-attack
## Dataset
We collect a single-label dataset and  merge instances from different websites at different overlapping rates. The settings of overlapping rate are the same as TMWF.

overlapping dataset: https://drive.google.com/drive/folders/1BRK21au25DiLJCjna0KfiOfzmQMZsmiL?usp=sharing

## Model
The "maincode" directory contains the prototype of our model.

~~~
maincode/
│
├── Util/ 
│   ├── Dataloader.py # The "dataloader.py" is used to generate the train and test dataloaders
│   └── utils.py # The "utils.py" contains the functions that are used by our model, such as the function that generates Q, K, and V used in 
│                  multi-head attention
│
├── net/
│   ├── TransformerDecoder/
│   │   ├── Encoder.py 
│   │   └── BertLayer.py
│   └── Encoder/
│
└── dir3/
    ├── file4.ext
    └── file5.ext
~~~
