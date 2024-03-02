# Toxic-Comment-Classification-LJMU
## Toxic Comment Classification LJMU Masters
In this project, we are carrying out the toxic comment classification using the Kaggle Toxic comment dataset, as a part of the MS with LJMU in the area of Artificial Intelligence and Machine Learning. 
We perform different experiments with three BERT variants and try to come up with a benchmark BERT fine tuned model for Toxic Comment Classification. 
Dataset: Dataset contains 
Target Columns: There are seven output columns that needs to be predicted by the model which denote different types of toxicity
  * Toxic
  * Severe_Toxic
  * Obscene
  * Threat
  * Insult
  * Identity_Hate
  * sexual_explicit

This problem is an example of Multi Label text classification where one record may belong to more than one class unlike multi class classification where the one record can belong to only one class. 

Models tried:

* BERT base
* BERT base with initial layers frozen
* RoBERTa
* DistilBERT

Hyper-Parameters:
* Batch Size
* Learning Rate

Loss Function:
* BCEwithLogitsLoss (With and without weights, pos_weight to handle the class imbalance issue)
* FocalLoss

Optimizer
Adam
AdamW
