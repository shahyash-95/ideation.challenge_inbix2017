# ideation.challenge_inbix2017
Inbix Ideation Challenge 

3D domain swapping is a mechanism by which two or more protein molecules form a dimer or higher oligomer by exchanging an identical structural element. This phenomenon is observed in an array of protein structures in oligomeric conformation. Protein structures in swapped conformations perform diverse functional roles and are also associated with deposition diseases in humans. 3D domain swapping problem was to predict and classify proteins into swapping or non swapping proteins based on sequence and structural features.

Dependenices:
1. sci-kit learn 
2. numpy

Feature Selection is done in 2 stages. 
1. Recursive feature elimination technique to identify the number of features to be selected in the model.
2. SelectKbest is a filter based method to select best features for the model.


Model Training
Neural Network model is trained with relu activation function.
Results are reported using roc curve and classification report.
