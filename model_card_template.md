# Model Card

Income Classification Model

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details - 

This model is a **Random Forest Classifier** trained to predict whether a person earns more than $50,000 per year based on features from the U.S. Census Adult Income dataset. The model was developed as part of a machine learning deployment pipeline using FastAPI and is designed to demonstrate best practices in MLOps, including model evaluation, deployment, and data slice analysis.

## Intended Use

The model is intended for **educational purposes and demonstration** of a production-grade ML pipeline. It classifies individuals as earning either `>50K` or `<=50K` based on demographic and occupational features.

## Training Data

The training dataset is derived from the U.S. Census Adult Income dataset. The features used include both categorical and numerical variables such as:

- Age
- Workclass
- Education
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Native Country
- Capital Gain/Loss
- Hours per Week

The categorical features were encoded using **OneHotEncoder**, and the target label (`salary`) was binarized using **LabelBinarizer**.


## Evaluation Data

20% of the full dataset was reserved for evaluation (test set) using stratified sampling to maintain class balance. The model was evaluated on this test set after training.

## Metrics

The model was evaluated using **Precision**, **Recall**, and **F1 Score**. The overall performance on the test set is:

- **Precision**: `0.7419`
- **Recall**: `0.6384`
- **F1 Score**: `0.6863`

## Ethical Considerations

- **Bias**: The dataset contains sensitive attributes (e.g., race, sex), and the model may inherit societal biases reflected in the data.
- **Fairness**: Disparities in performance across demographic slices were reviewed, but no bias mitigation strategies were applied in this baseline model.
- **Usage Warning**: This model is not suited for real-world deployment involving employment, finance, or housing decisions.


## Caveats and Recommendations

- The current model does not perform hyperparameter tuning and may benefit from optimization.
- Results may vary if applied to different populations or time periods.
- Further steps should be taken to mitigate bias, ensure fairness, and secure data privacy for production use.
- For production readiness, consider:
  - Monitoring model drift
  - Logging requests and responses
  - Applying explainability tools (e.g., SHAP)