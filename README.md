# AMPGP
    # Data folder: contains data for generating model and forecasting model.
      # Generate model data: CAMP (two formats)
      # Data of prediction model: AMP (training set and independent test set)
      # Non-AMP (training set and independent test set)
    # Feature folder: contains two methods of feature selection (relief method and mutual information method)
      # Feature_fuse.py refers to the realization of one-step feature fusion in our final model.
      # Label.csv: refers to all labels that generate the training data of the model.
      # Normalized_combine_all.csv: It is a table transformation in which all 16 feature data are put together.
      # Combine_AAC_PAAC_Qsorder.csv: three sequence information feature data after feature screening.
      # Combine _ aaindex _ gaac _ ctdd _ ctdc.csv: four characteristic data of physical and chemical information after screening.
      # Combine_BLOUSM62_PSSM.csv: Two evolutionary information feature data after feature selection.
    # Model folder: Parameter data of the established models after training.
 
    # Attention.py
    # BiLSTM.py
    # BiLSTMAttention.py
    # CNN.py
    # CNNAttention.py
    # CNNBiLSTM.py
    # CNNBiLSTMAttention.py
    # These seven functions are the established deep learning model. In the end, the most suitable feature is selected for each type, and the bold four types are finally selected, or only these four types can be copied.
    # Generate_model.py: code to generate the model.
    # Predict_model.py: the final code of our prediction model.
