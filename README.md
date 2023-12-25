## Predicting Purchases from Internet Usage Patterns
- This project draws inspiration from a [Harvard CS50 Course](https://cs50.harvard.edu/ai/2023/projects/4/shopping/). The dataset contains information on users' internet usage patterns, including the types of webpages visited, the duration spent on each, and Google Analytics metrics such as bounce rates, exit rates, and page values. Additionally, it includes data on whether a user made a purchase during a session.
- This binary classification problem was approached in two ways - using a Random Forest Classifier and using a neural network. Given a significant class imbalance, where the majority class had nearly five times as much training data as the minority class, our goal was to achieve high specificity and sensitivity. Hence, Youden Index was chosen as a metric to assess the model's performance.
### Using Random Forest Classifier:
- The dataset was pre-processed using 'csv.reader'. To address the high class imbalance, different data sampling techniques such as Random Under Sampling, Random Over Sampling and SMOTE Sampling were tried and among these, Random Under Sampling yielded the best results.
- Notably, Random Forest Classifier outperformed Gradient Boosting Classifier, achieving an 83% sensitivity, 86.6% specificity and a Youden Index of 0.697.

### Using a Neural Network:
- The dataset was pre-processed using Pandas. Random Over Sampling was chosen as the desired sampling method to avoid data loss with Random Under Sampling.
- A custom metric function was used to calculate the Youden Index and that was used as a metric during model compilation. 
- Even after data sampling, assigning 1.2 times the weightage to the minority class during model fitting slightly improved the performance.
- Careful hypertuning led to achieving about 82% sensitivity, 84% specificity and a Youden Index of about 0.66.

It can be observed that the Random Forest Classifier performed slightly better than the neural network. The difference in performance could be attributed to the limited data available for the minority class.
