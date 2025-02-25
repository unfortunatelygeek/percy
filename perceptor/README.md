# Perceptor: Intelligent Test Execution Prediction within CI/CD pipeline and Intelligent Documentation

The primary goal of Perceptor is to predict which test cases are necessary for a given code change. This helps in optimizing the testing process by running only the relevant tests, thereby reducing execution time and resources. Perceptor also generates documentation based on the changelog, the added code and gitlog of the codebase. It uses a local LLM to do this, for security purposes.

## How the SVM in Perceptor Works:

- **Data Extraction**: Perceptor extracts CI/CD workflows from .github/workflows files. These workflows contain information about the build, test, and deployment processes.

- **Feature Engineering**: Relevant features are extracted from these workflows and the code changes.

- **SVM Model Training**: An SVM classifier is trained on these features to learn patterns that indicate which tests are likely to fail or require execution based on specific code changes.

- **Prediction**: When new code changes are introduced, the trained SVM model predicts which test cases should be executed. This prediction is based on the patterns learned during training.

## Why Do This?
Efficiency: By predicting and executing only necessary tests, Perceptor reduces the overall testing time and resource usage, making the CI/CD process more efficient.

## Challenges and Considerations:
- **Data Quality**: The effectiveness of the SVM model depends on the quality and relevance of the data used for training. Ensuring that the features extracted are meaningful and representative is crucial.
- **Model Maintenance**: As the codebase evolves, the SVM model may need periodic retraining to maintain its accuracy and adapt to new patterns in code changes.