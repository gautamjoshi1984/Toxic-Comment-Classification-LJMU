# Toxic-Comment-Classification-LJMU
### Toxic Comment Classification LJMU Masters
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

### Models tried:

* BERT base
* BERT base with initial layers frozen
* RoBERTa
* DistilBERT

### Hyper-Parameters:
* Batch Size
* Learning Rate

### Loss Function:
* BCEwithLogitsLoss (With and without weights, pos_weight to handle the class imbalance issue)
* FocalLoss
* Dice Loss
* Focal Tversky Loss

### Optimizer
* AdamW

# Results
Training Use case									Macro				Micro			
Model Name	Number of Classes	Batch Size	Learning Rate	Epochs	Loss Function	Dropout	BERT Layers frozen	Loss Function Parameter	Validation AUCROC	Validation Recall	Validation Precision	Validation F1	Validation AUCROC	Validation Recall	Validation Precision	Validation F1
bert-base-uncased	7	128	1*e-05	7	BCEWithLogitsLoss	0.3	No	NA	0.8261	0.691	0.7252	0.6556	0.9298	0.8882	0.8268	0.8564
bert-base-uncased	7	64	2*e-05	6	BCEWithLogitsLoss	0.2	No	NA	0.825	0.689	0.694	0.662	0.93	0.889	0.829	0.858
bert-base-uncased	7	64	1*e-05	6	BCEWithLogitsLoss	0.2	No	NA	0.832	0.708	0.75	0.654	0.935	0.903	0.812	0.855
bert-base-uncased	7	32	1*e-05	6	BCEWithLogitsLoss	0.3	Yes	NA	0.83	0.701	0.761	0.658	0.936	0.902	0.819	0.858
bert-base-uncased	7	32	1*e-05	6	BCEWithLogitsLoss	0.3	No	Weights	0.8329	0.7102	0.6403	0.6566	0.9339	0.9008	0.8076	0.8517
bert-base-uncased	7	32	1*e-05	6	BCEWithLogitsLoss	0.3	No	POS_weight	0.8797	0.7957	0.5499	0.6244	0.8856	0.8054	0.784	0.7945
bert-base-uncased	6	128	1*e-05	5	BCEWithLogitsLoss	0.3	No	NA	0.886	0.82	0.702	0.756	0.939	0.913	0.82	0.864
bert-base-uncased	6	64	2*e-05	5	BCEWithLogitsLoss	0.3	No	NA	0.895	0.835	0.7	0.759	0.936	0.908	0.824	0.864
bert-base-uncased	6	64	1*e-05	5	BCEWithLogitsLoss	0.3	No	NA	0.878	0.81	0.716	0.757	0.94	0.921	0.804	0.859
bert-base-uncased	6	32	1*e-05	5	BCEWithLogitsLoss	0.3	No	NA	0.883	0.814	0.704	0.753	0.938	0.913	0.816	0.861
bert-base-uncased	6	32	1*e-05	5	BCEWithLogitsLoss	0.3	Yes	NA	0.869	0.79	0.724	0.749	0.94	0.92	0.809	0.861
bert-base-uncased	6	32	1*e-05	5	BCEWithLogitsLoss	0.3	No	Weights	0.931	0.901	0.617	0.712	0.913	0.862	0.814	0.837
bert-base-uncased	6	32	1*e-05	6	BCEWithLogitsLoss	0.3	No	POS_weight = 5	0.927	0.95	0.555	0.693	0.952	0.98	0.699	0.816
bert-base-uncased	6	32	1*e-05	5	Focal Loss	0.4	No	gamma = 2, POS_weight	0.934	0.918	0.608	0.714	0.929	0.901	0.789	0.841
bert-base-uncased	6	32	1*e-05	5	Focal Loss	0.4	No	gamma = 2, POS_weight = weights	0.934	0.913	0.581	0.677	0.911	0.864	0.789	0.825
bert-base-uncased	6	32	1*e-05	5	Focal Loss	0.4	No	gamma = 3, POS_weight = 15	0.931	0.956	0.563	0.7	0.952	0.978	0.705	0.819
RoBERTa	7	32	1*e-05	6	BCEWithLogitsLoss	0.3	No	None	0.832	0.7	0.684	0.662	0.928	0.834	0.883	0.858
RoBERTa	7	32	1*e-05	6	BCEWithLogitsLoss	0.3	No	POS_weight	0.883	0.808	0.555	0.643	0.906	0.849	0.78	0.813
RoBERTa	7	32	1*e-05	6	BCEWithLogitsLoss	0.3	No	Weights	0.831	0.7	0.659	0.663	0.93	0.889	0.827	0.857
RoBERTa	6	32	1*e-05	5	Focal Loss	0.4	No	Gamma  = 5, pos_weight = 40	0.918	0.976	0.473	0.625	0.94	0.991	0.617	0.76
RoBERTa	6	64	1*e-05	5	Focal Loss	0.3	No	Gamma  = 2, pos_weight = 30	0.927	0.968	0.505	0.653	0.948	0.986	0.792	0.792
RoBERTa	7	32	1*e-05	6	Focal Loss	0.3	No	Gamma = 2, POS_Weight	0.893	0.844	0.521	0.611	0.899	0.855	0.7	0.77
RoBERTa	7	32	1*e-05	6	Focal Loss	0.3	No	Gamma = 3, POS_Weight	0.884	0.814	0.523	0.613	0.913	0.867	0.762	0.811
RoBERTa	6	32	1*e-05	6	Focal Loss	0.3	No	Gamma = 2, POS_Weight	0.933	0.907	0.606	0.701	0.913	0.864	0.806	0.834
RoBERTa	6	32	1*e-05	6	Focal Loss	0.3	No	Gamma = 2, POS_Weight = Custom weights	0.939	0.921	0.6	0.703	0.918	0.875	0.8	0.835
RoBERTa	6	32	1*e-05	6	Focal Loss	0.4	No	Gamma = 2, POS_Weight = Custom weights	0.941	0.934	0.583	0.695	0.927	0.898	0.781	0.836
RoBERTa	6	32	1*e-05	6	Focal Loss	0.2	No	Gamma = 2, POS_Weight = Scalar 10	0.931	0.963	0.538	0.679	0.951	0.983	0.689	0.81
RoBERTa	6	32	1*e-05	5	Focal Loss	0.1	No	Gamma = 5, POS_Weight = Scalar 20	0.93	0.966	0.53	0.675	0.95	0.985	0.679	0.804
RoBERTa	6	32	1*e-05	5	Focal Loss	0.2	No	Gamma = 2, POS_Weight = Scalar 8	0.933	0.963	0.548	0.689	0.953	0.981	0.7	0.817
RoBERTa	6	32	1*e-05	5	Focal Loss	0.4	No	Gamma = 3, POS_Weight = Scalar 15	0.932	0.962	0.552	0.692	0.952	0.981	0.695	0.813
RoBERTa	6	32	1*e-05	6	Focal Loss	0.4	Yes	Gamma = 2, POS_Weight = Custom weights	0.936	0.915	0.609	0.711	0.922	0.882	0.805	0.841
RoBERTa	6	32	1*e-05	6	Focal Loss	0	No	Gamma = 2, POS_Weight = Custom weights	0.937	0.917	0.584	0.68	0.904	0.848	0.795	0.821
RoBERTa	6	32	1*e-04	5	Focal Loss	0.4	No	Gamma = 2, POS_Weight = Custom weights	0.925	0.906	0.544	0.64	0.903	0.857	0.75	0.8
RoBERTa	6	32	1*e-05	5	Dice Loss	0.3	No	Weights	0.605	0.245	0.235	0.24	0.864	0.752	0.826	0.787
DistilBERT	7	32	1*e-05	6	BCEWithLogitsLoss	0.2	No	None	0.815	0.667	0.771	0.648	0.93	0.887	0.834	0.86
DistilBERT	6	32	1*e-05	5	BCEWithLogitsLoss	0.3	No	None	0.87	0.79	0.728	0.755	0.942	0.92	0.818	0.866
DistilBERT	7	32	1*e-05	6	BCEWithLogitsLoss	0.2	No	POS_weight	0.882	0.802	0.542	0.621	0.893	0.821	0.78	0.8
DistilBERT	7	32	1*e-05	6	BCEWithLogitsLoss	0.2	No	POS_weight with scalar 5	0.861	0.787	0.56	0.646	0.948	0.947	0.745	0.834
DistilBERT	6	32	1*e-05	4	BCEWithLogitsLoss	0.1	No	POS_weight with scalar 10	0.925	0.94	0.575	0.708	0.952	0.974	0.715	0.825
DistilBERT	6	32	1*e-05	4	BCEWithLogitsLoss	0.4	No	POS_weight with scalar 5	0.92	0.928	0.583	0.708	0.951	0.971	0.72	0.826
DistilBERT	6	32	1*e-05	4	BCEWithLogitsLoss	0.1	No	POS_weight with scalar 10	0.927	0.932	0.597	0.722	0.952	0.965	0.741	0.838
DistilBERT	6	32	1*e-05	6	BCEWithLogitsLoss	0.4	No	Weights	0.871	0.78	0.727	0.751	0.924	0.877	0.843	0.859
DistilBERT	6	32	1*e-05	6	Focal Loss	0.3	No	Gamma = 2, POS_Weight	0.922	0.877	0.612	0.686	0.878	0.787	0.818	0.802
DistilBERT	6	32	1*e-05	5	Focal Loss	0.4	No	Gamma = 2, POS_Weight = weights	0.925	0.889	0.596	0.68	0.893	0.822	0.807	0.815
DistilBERT	6	32	1*e-05	5	Focal Loss	0.4	No	Gamma = 2, POS_Weight = Scalar 10	0.926	0.946	0.568	0.704	0.952	0.978	0.705	0.819
DistilBERT	6	32	1*e-05	5	Focal Loss	0.4	No	Gamma = 2, POS_Weight = Scalar 15	0.927	0.95	0.551	0.689	0.951	0.978	0.7	0.816
DistilBERT	6	32	1*e-05	5	Focal Loss	0.4	No	Gamma = 3, POS_Weight = Scalar 15	0.927	0.953	0.558	0.696	0.951	0.98	0.695	0.813
![image](https://github.com/gautamjoshi1984/Toxic-Comment-Classification-LJMU/assets/28692811/5869a82c-f85c-44b4-aee2-0395cafcc5b8)
