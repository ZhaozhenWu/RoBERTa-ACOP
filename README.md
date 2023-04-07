# RoBERTa-ACOP
RoBERTa-ACOP implementation  
* Code and preprocessed dataset for paper titled "Multitasking for Aspect-based Sentiment Analysis via Constructing Auxiliary Self-Supervision ACOP task"  
* Zhaozhen Wu  
  
# Requirements  
* Python 3.8.12
* PyTorch 1.7.1
* numpy 1.21.2
* transformers 3.5.0
* spacy 3.1.3  
  
# Usage  
* Install SpaCy package and language models with   
```pip install spacy``` 
and   
```python -m spacy download en```
* Install PyTorch with   
```pip install spacy``` 
* Install transformers with   
```pip install transformers```
* Generate dataset of ACOP task with  
```python generateOrder.py```
* Download pretrained RoBERTa with this **[link](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta)**
* Train with command, optional arguments could be found in train.py
  
# Model  
We propose a auxiliary task called Aspect and Context Order Prediction (ACOP) for Aspect-based Sentiment Analysis and we construct a new model called RoBERTa-ACOP by integrate our auxiliary task into RoBERTa model with multitasking way.  
  
An overview of our proposed model is given below  
<img src="https://user-images.githubusercontent.com/52657545/230539400-76b0eac1-d1ae-4d9e-a354-d4bb005edaf5.png" width="800" height="1200"/><br/>
  
# Note  
* Code of this repo heavily relies on **[ASGCN](https://github.com/GeneZC/ASGCN)**
