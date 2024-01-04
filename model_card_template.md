# Model Card


## Model Details
    Model Date: January 5,`` 2024
    Model Version: 1.0
    Model Type: Classification
    Developers: Panagiotis Stavrianos
    Model Description: This model is designed to classify the salary of each individual feature characteristics

## Intended Use

    Primary Intended Uses: This model is intended for classifying human salary range.
    Primary Intended Users: Data scientists and end-users interested in recognizing salary from characteristics

Factors

    Evaluation Factors: The model has been evaluated in several data slices for each categorical feature.

## Training Data

    Datasets: Census Dataset, containing ~32K labelled samples.
    Motivation: This dataset was chosen for its diversity.
    Preprocessing: Encoder and label binarizer are trained through offline training and employed from the last one.

## Evaluation Data
 
    Datasets: Census Dataset, containing ~32K labelled samples.
    Motivation: This dataset was chosen for its diversity.
    Preprocessing: Encoder and label binarizer are trained through offline training and employed from the last one.

## Metrics

    Model Performance Measures: Precision, Recall and F1 Score around data slices in slice_output.txt.
    Decision Thresholds: The model uses a threshold of 0.5 for classifying an object into a specific category.

## Ethical Considerations

    Ethical Risks: Potential for biased performance in certain underrepresented categories.
    Mitigation Strategies: Regular updates and retraining with more diverse datasets.

## Caveats and Recommendations

    Known Limitations: The model may have reduced accuracy as it is not trained specifically for imbalanced cases.
    Recommendations: Adviced to push development for imbalanced, yet this demonstrated only API utilities.