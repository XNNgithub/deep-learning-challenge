# deep-learning-challenge
# Instructions

# Step 1: Preprocess the Data
Using your knowledge of Pandas and scikit-learn’s `StandardScaler()`, you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.
Using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.
1. Read in the `charity_data.csv` to a Pandas DataFrame, and be sure to identify the following in your dataset:
- What variable(s) are the target(s) for your model?
- What variable(s) are the feature(s) for your model?
2. Drop the `EIN`and `NAME` columns.
3. Determine the number of unique values for each column.
4. For columns that have more than 10 unique values, determine the number of data points for each unique value.
5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then check if the binning was successful.
6. Use `pd.get_dummies()` to encode categorical variables.
7. Split the preprocessed data into a features array, `X `, and a target array, `y `. Use these arrays and the `train_test_split `function to split the data into training and testing datasets.
8. Scale the training and testing features datasets by creating a `StandardScaler` instance, fitting it to the training data, then using the `transform `function.

# Step 2: Compile, Train, and Evaluate the Model
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organisation will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.
1. Continue using the Jupyter Notebook in which you performed the preprocessing steps from Step 1.
2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
3. Create the first hidden layer and choose an appropriate activation function.
4. If necessary, add a second hidden layer with an appropriate activation function.
5. Create an output layer with an appropriate activation function.
6. Check the structure of the model.
7. Compile and train the model.
8. Create a callback that saves the model's weights every five epochs.
9. Evaluate the model using the test data to determine the loss and accuracy.
10. Save and export your results to an HDF5 file. Name the file `AlphabetSoupCharity.h5.`

# Step 3: Optimise the Model
Using your knowledge of TensorFlow, optimise your model to achieve a target predictive accuracy higher than 75%.
Use any or all of the following methods to optimise your model:
- Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
    - Dropping more or fewer columns.
    - Creating more bins for rare occurrences in columns.
    - Increasing or decreasing the number of values for each bin.
- Add more neurons to a hidden layer.
- Add more hidden layers.
- Use different activation functions for the hidden layers.
- Add or reduce the number of epochs to the training regimen.
Note: If you make at least three attempts at optimising your model, you will not lose points if your model does not achieve target performance.
1. Create a new Jupyter Notebook file and name it `AlphabetSoupCharity_Optimisation.ipynb.`
2. Import your dependencies and read in the `charity_data.csv `to a Pandas DataFrame.
3. Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimising the model.
4. Design a neural network model, and be sure to adjust for modifications that will optimise the model to achieve higher than 75% accuracy.
5. Save and export your results to an HDF5 file. Name the file `AlphabetSoupCharity_Optimisation.h5. `

# Step 4: Write a Report on the Neural Network Model
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.
The report should contain the following:
1. Overview of the analysis: Explain the purpose of this analysis.
2. Results: Using bulleted lists and images to support your answers, address the following questions:
- Data Preprocessing
    - What variable(s) are the target(s) for your model?
    - What variable(s) are the features for your model?
    - What variable(s) should be removed from the input data because they are neither targets nor features?
- Compiling, Training, and Evaluating the Model
    - How many neurons, layers, and activation functions did you select for your neural network model, and why?
    - Were you able to achieve the target model performance?
    - What steps did you take in your attempts to increase model performance?
3. Summary: Summarise the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.

# Write a Report on the Neural Network Model
Data Preprocessing
1. What variable(s) are the target(s) for your model? 
'IS_SUCCESSFUL' column from application_df
2. What variable(s) are the features for your model?
Every other column from application_df
3. What variable(s) should be removed from the input data because they are neither targets nor features?
Both 'EIN' and 'NAME' columns were dropped/removed, because they were neither targets nor features for the dataset

Compiling, Training, and Evaluating the Model
1. How many neurons, layers, and activation functions did you select for your neural network model, and why?
In the first attempt, I used 8 hidden_nodes_layer1 and 5 hidden_nodes_layer2, which were essentially random selections that served as starting points for refinement in the second iteration.
2. Were you able to achieve the target model performance?
I was not able to achieve the 75% model accuracy target.
3. What steps did you take in your attempts to increase model performance?
I increased the number of layers, removed additional columns, added extra hidden nodes, and altered the activation functions for each layer, all in an effort to enhance the model's accuracy.

In summary, the deep learning model achieved a 73% accuracy rate for the classification task. Improving the correlation between input and output is expected to enhance prediction accuracy. This improvement can be accomplished by performing initial data cleanup and experimenting with models featuring different activation functions, iterating until a higher accuracy level is attained.
