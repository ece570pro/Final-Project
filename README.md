# Final-Project

# Thermal Conductivity Prediction of 2D Woven Composites

This repository contains the codebase for a data-driven framework that predicts the effective thermal conductivity of 2D woven composites using a two-step MSG-based homogenization process and artificial neural networks (ANNs). It includes scripts for dataset generation via Latin Hypercube Sampling (LHS), MSG based homogenization, ANN model training, and optional deployment as an API for integration with natural language platforms like OpenAI’s Custom GPT.

---

To run the ANN model , you only need the following python packages installed :

1. Tesnorflow
2. Pandas
3. numpy
4. scikit-learn
5. matplotlib

   Use the command : pip install tensorflow pandas numpy scikit-learn matplotlib


If you want to run the MSG based scripts for homogenization (which is just for input data generation), you will need additional softwares as indicated in the "Project Workflow and Requirements" document in the Reference Paper folder. 

# Running the ANN model
1. Go to the directory with FinalANN.py
2. Ensure all requuired moodules are installed
3. Run the script - python FinalANN.py
4. To plot the results - python Plots.py
5. To make new predictions - python Predictions.py

