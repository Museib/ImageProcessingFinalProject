User Manual for my Image Processing Final Project
===============================
1. Install Dependencies
pip install tensorflow keras-tuner scikit-learn matplotlib seaborn

	1.1 Install and uncomment necessary line if need to perform hyperparameter search:
	Line 5: 
		# !pip install -q keras-tuner
	Line 201 and lines below : 
		# Uncomment below lines to perform the hyperparameter search
		# tuner.search(
		...

2. Run the Script
Make sure you are in an environment with the above libraries installed.
Execute the script

3. Hyperparameter Tuning (via both grid search and random search using Keras Tuner).
By default, the code runs a final model with fixed hyperparameters.
To perform hyperparameter tuning with Keras Tuner:
	Uncomment the tuner.search(...) lines in the code.
	Adjust max_trials, epochs, and patience as needed for your search.
	After the search completes, fetch the best hyperparameters with tuner.get_best_hyperparameters().
	Build and train your model again using the best hyperparameters to get the final model.
	
4. Optional Transfer Learning
If you want to try transfer learning (e.g., with MobileNetV2), uncomment the relevant section and replace the model in your training pipeline with build_transfer_model(...).
Make sure to match input dimensions to what the pre-trained model expects (e.g., 224x224 for MobileNetV2).
Potentially change the data preprocessing from [0,1] normalization to the appropriate preprocessing function (e.g., mobilenet_v2.preprocess_input).

===============================

This pipeline satisfies the requirements of:

Dataset Preparation & Preprocessing (including normalization, data splits, and augmentation).
Model Development (custom CNN + optional transfer learning).
Regularization (dropout, L2, early stopping).
Hyperparameter Tuning (via Keras Tuner’s random search).
Model Evaluation (accuracy, confusion matrix, classification report, and training/validation curves).

================================

Note:

This code uses TensorFlow (>=2.7) and Keras Tuner (>=1.1).
If you want to adapt it to other datasets (e.g., Fashion-MNIST, or a custom Kaggle dataset), replace the data-loading portion accordingly.
