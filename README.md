# RoBERTa-ACOP
RoBERTa-ACOP implementation  
* Code and preprocessed dataset for paper titled "Multitasking for Aspect-based Sentiment Analysis via Constructing Auxiliary Self-Supervision ACOP task"  
* Zhaozhen Wu  
  
# Requirements  
* Python 3.8
* PyTorch 1.0.0
* numpy 1.15.4  
  
# Usage  
* Install SpaCy package and language models with   
```pip install numpy```  
  
# Model  
We propose a auxiliary task called Aspect and Context Order Prediction (ACOP) for Aspect-based Sentiment Analysis and we construct a new model called RoBERTa-ACOP by integrate our auxiliary task into RoBERTa model with multitasking way.  
  
An overview of our proposed model is given below  
![image](https://user-images.githubusercontent.com/52657545/195820065-45379337-2376-4e79-b561-302605b602de.png)
