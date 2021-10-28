# GOAT
#### Here is the source code for the article 'Ready for emerging threats to recommender systems? A graph convolution-based generative shilling attack' published on Information Sciences.<br>

#### The primary model (GOAT) is achieved by TensorFlow in version 1.0; another version of TensorFlow 2.0 will be offered in the future.<br>

## DouBan_small
#### We offered a small dataset extracted from DouBan to help to check whether the code is runnable.<br>

## Dao_atk.py
#### The data-processing module. <br>

## AttackGenerator_xx.py
#### Shilling attack model initializing and training. <br>

## GenerateFromModel.py
#### Generate fake profiles after the model was saved. <br>

## Other tips
#### *How to use it: (in command line) 'python AttackGenerator_xx.py'
#### *It would take a while at the first running for data processing, and the results will be saved locally.
#### *The final generated ratings do not contain attack targets, please add them manually if needed.
#### Further questions, please contact the corresponding author.
