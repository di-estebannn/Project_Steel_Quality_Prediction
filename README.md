# Project Submission Template

## 📁 Repository Structure
Below is an example repository structure to help you to navigate through the various segments and files for your chosen project:

- `/project_steel_quality_prediction/`
    - `src/`
        - `notebooks/`
            - `process_data.ipynb`
            - `visualize_data.ipynb`
            - `models.ipynb`
        - `scripts/`
            - `preprocess_data.py`
            - `graph.py`
            - `train_model.py`
    - `data/`
        - `raw/`
            - `normalized_train_data.csv`
            - `normalized_test_data.csv`
            - `normalized_total_data.csv`
        - `processed/`
            - `processed_total_data.csv`
            - `processed_training_data.csv`
            - `processed_testing_data.csv`
            - `processed_validation_data.csv`
    - `docs/`
        - `Report.pdf`
        - `Presentation.pdf`
    - `results/`
        - `figures/`
            - `box_plots_raw_t.png`
            - `box_plots_processed_t.png`
            - `scatter_plots_raw_t.png`
            - `scatter_plots_processed_t.png`
            - `scatter_plots_regression_raw.png`
            - `scatter_plots_regression_processed.png`
            - `heat_maps_raw_t.png`
            - `heat_maps_processed_t.png`
            - `histograms_raw_t.png`
            - `histograms_processed_t.png`
            - `feature_importance_multiple_regression.png`
            - `feature_importance_random_forest.png`
            - `learning_curve_multiple_regression.png`
            - `learning_curve_random_forest.png`
            - `learning_curve_svm.png`
            - `learning_curve_nn.png`
        - `tables/`
            - `statistics_of_raw_total_data.csv`
            - `statistics_of_processed_total_data.csv`
            - `model_performance_results.csv`
        - `models/`
            - `multiple_regression.pkl`
            - `random_forest_regressor.pkl`
            - `support_vector_regressor.pkl`
            - `neural_network_regressor.pkl`
    - `README.md`

**Explanation of the structure:**

- `src/`: Contains all the source code files.
- `data/`: Contains all the data used in the project.
- `docs/`: Contains documentation (report and presentation).
- `results/`: Contains the outputs of the analyses.
- `README.md`: This file, containing information about the project.

## Project Report Format

### Project Title

Application and comparison of Machine Learning Algorithms for Steel Quality Prediction

### Abstract

The following project aims to present the application and comparison of machine learning algorithms for steel quality prediction in continuous casting plants with data from the ‘Stahl- und Walzwerk Marienhütte GmbH Graz’. Therefore, given a dataset of sensor and process data, different models will be designed based on Machine Learning Algorithms to predict a quality relevant metric.

### Introduction

**Objectives**

     General objective:
Predict a quality relevant metric (yield strength) through the use of different machine learning algorithms, training the models with the given dataset of sensor and process data.

     Specific objectives:
● Implement effective data set cleaning techniques and save the processed data.
● Make and sabe graphs to visualize the distribution and trends of the data.
● Perform analysis of the data set based on its statistics and graphs.
● Develop machine learning models to predict steel quality from the continuous casting plants dataset provided.
● Train the model to learn and map the input process data to the target steel quality variables.
● Compare various ML architectures to identify the most effective model for accurate steel quality prediction.
● Assess the performance

**Background**

Artificial Intelligence (AI) and Machine Learning (ML) have evolved significantly in recent decades due to advances in algorithms, massive data availability and computational power. ML, an essential branch of AI, has proven to be fundamental in allowing machines to learn patterns and make decisions without human intervention, transforming the way we approach problems and optimize processes in all types of industries, areas and environments, like from health to production.

Its importance lies in the ability to automate complex tasks, perform predictive analysis and offer efficient solutions, promoting innovation and improving decision making in real time. Thus, within the framework of the application of machine and deep learning techniques, this project focuses on the creation and optimization of predictive models to predict the quality of steel in continuous casting plants, based on a data set with 21 sensor and process data as influential variables in the creation of steel and its final quality result.

During the development of the project, various data preprocessing techniques were applied, such as separating the original data set into subsets with different functions, cleaning null values and their management to avoid losing information, and removing outliers that may influence the training of the model, in order to guarantee the quality and consistency of the information. Subsequently, Cross Validation and supervised learning algorithms were used to build predictive models, while the selection of hyperparameters was carried out using the Grid Search technique.

This project seeks not only to provide efficient solutions for predicting the quality of the steel produced, but also to lay the foundations for the application of similar methodologies in industrial and production contexts, promoting the effective implementation of machine learning techniques for the continuous improvement of processes and informed decision making, thus contributing to the improvement of efficiency and quality in industrial processes.

## Methods

**Access and prepare (preprocess and cleaning) the data for Machine Learning tasks:**

Data is loaded from CSV files corresponding to normalized training and test sets. The data is then combined and separated in training, testing and validation sets, applying preprocessing functions, such as shuffle_data and separate_and_clean_data, transforming all values into numeric (or null if it cannot be numeric) and dropping any row with null outputs. In addition, the resulting total data is saved in a new CSV file named "normalized_total_data.csv".de humedad.

Later, the separate_and_clean_data function performs two main tasks. First, if the data is not pre-cleaned, it converts all values in the data set to the numeric type, assigning null values to those that cannot be converted. Then, it removes rows that have null values in the 'output' column and removes duplicates from the data set. Second, the function separates the data set into inputs (X) and outputs (y), and returns these two parts. Additionally, it prints the sizes of the resulting arrays, providing information about the structure of the clean, separated data. This process is essential to prepare data before using it in training machine and deep learning models.

Then, the fill_null_values function is responsible for handling null values in the data set. First, it concatenates the inputs (X) and outputs (y) into a single DataFrame (df). Next, check if a set of means is provided. If not provided, calculates the means of the input columns (X). Next, replace the null values in X with the calculated means. Only with the testing set a set of means is provided, corresponding to the means of the training set so that the first one is effectively related and based on the second one, while this is not done with the validation set so as not to influence the next validation that will be done of the effectiveness of the trained model.
The function prints descriptive statistics of the data set, highlighting the total amount of data in all columns and whether null values are present. It also shows the means of the 21 inputs in tabular form.

The final step for processing data is to remove outliers. The remove_outliers function identifies and removes outliers in the data set. First, it concatenates the outputs (y) and inputs (X) into a single DataFrame (df). It then iterates over the input columns and uses the interquartile range (IQR) method to determine the upper and lower bounds, based on a default threshold = 3.75. This is done 5 times to ensure removal of outliers that are outside the set range each time it is updated. Values that fall outside these limits are considered outliers and are removed. The function prints the number of rows removed and the percentage of data removed from the original set. Importantly, the function does not remove outliers in the validation set (X_validation, y_validation) in order not to influence the next validation that will be done of the effectiveness of the trained model. This process helps improve the robustness of the model by removing data that could introduce noise or bias during training.

Finally, the save_data function takes the input data set (X and y) and saves them to CSV files with the specified name. If the data is not instances of pd.Series or pd.DataFrame, it converts it to those formats. Then, concatenate (y) and (X) into a single DataFrame (combined_df), sort the columns and rename the input columns as "input1", "input2", ..., "input21", and the output column as "output". Finally, save the DataFrame to a CSV file at the specified path (data_path) with the given file name (name_file). The function prints a message indicating that the file has been saved successfully, or displays an error message if a problem occurs during the process. In this case, the functions are used to save different processed data sets (training, testing, validation, and total) to specific CSV files.

**Tools Used**

import pandas as pd

import numpy as np

import os

## Results

**Findings**

The models with the lowest error are the Random Fortes and the Support Vector Machine. Furthermore, in their learning curve you could see that they were already stabilizing and more data would overfit them. However, although the Neural Network did not perform as well, it did have the potential to continue learning, but there was not enough data.
![learning_curve_multiple_regression](https://github.com/di-estebannn/Project_Steel_Quality_Prediction/assets/122711172/5aaed1d2-336b-44f7-83fd-89c0dbcbf9f5)

![learning_curve_random_forest](https://github.com/di-estebannn/Project_Steel_Quality_Prediction/assets/122711172/f088be42-e670-4745-9566-cfd8834bbb13)

![learning_curve_svm](https://github.com/di-estebannn/Project_Steel_Quality_Prediction/assets/122711172/12dc1fcc-7ad1-4058-9eaf-d7a7cec0f4db)

![learning_curve_nn](https://github.com/di-estebannn/Project_Steel_Quality_Prediction/assets/122711172/5550cf08-44fa-42dc-b9f5-1796371076ad)


**Visualizations**

![image](https://github.com/di-estebannn/Project_Steel_Quality_Prediction/assets/122711172/926311b9-12d3-4e89-b12c-8ccf6188c7d4)


Model,____________________Mean,_________Standard deviation,___MSE on Testing set,___MSE on Validation set,___Training duration

Multiple Regression,______0.00609323,___0.01857774,___________0.00628793,___________0.00648502,______________0.20045264

Random Forest,____________0.00414187,___0.02001575,___________0.00428681,___________0.00451971,______________42.4779955

Support Vector Machine,___0.00498062,___0.02146126,___________0.00524842,___________0.00509759,______________22.8429364

Neural Network,___________0.00597893,___0.02023769,___________0.00623839,___________0.00642575,______________94.8201293

## Conclusion

The results obtained from all machine learning models demonstrate sufficient efficiency and functionality. However, their differences, advantages, and disadvantages are apparent. In general terms, the current preferred choices are Random Forest as the top option, followed by Support Vector Machine. These models exhibited the smallest error both in the training dataset with Cross Validation and when tested with the testing and validation datasets.

Moreover, their learning curves performed well and displayed no indications of overfitting. Although the SVM curve appeared more favorable, the Random Forest is preferred due to its lower Mean Squared Error (MSE) and the additional benefit of providing access to the most relevant features, enabling further analysis.

On the other hand, if a substantial amount of additional data were to become available, the Neural Network might become the optimal choice. As evident from its learning curve, there is still potential for further improvement and error reduction. However, due to the limited amount of data available, the Neural Network did not achieve its full training potential.

Ultimately, the selection of one model over another depends on the specific circumstances and required conditions. It is crucial to have a clear understanding of the operation of each model, their hyperparameters, and also comprehend other phases of creating machine learning models, such as data preprocessing and information visualization.
