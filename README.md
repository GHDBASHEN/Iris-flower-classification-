# Iris Flower Classification

This project is a simple machine learning implementation for classifying Iris flowers into three species: **setosa**, **versicolor**, and **virginica**. It uses the popular **Iris dataset** from the UCI Machine Learning Repository.

## Project Overview
- **Objective**: Classify Iris flowers based on sepal and petal length and width.
- **Dataset**: The Iris dataset consists of 150 samples, each with four features:
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- **Target labels**: Three classes representing different flower species.

## Dependencies
To run this project, ensure the following Python libraries are installed:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Install them using:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## How to Run
1. Clone the repository or download the `iris_classification.py` file.
2. Run the script:
   ```bash
   python iris_classification.py
   ```
3. The program will:
   - Load and visualize the Iris dataset.
   - Split the data into training and testing sets.
   - Train a Logistic Regression model.
   - Evaluate the model using accuracy and a classification report.
   - Optionally, test a K-Nearest Neighbors (KNN) model.

## Example Output

The program will display a scatter plot of the Iris dataset and print the model's accuracy and classification report

![image](https://github.com/user-attachments/assets/5e6bf5ac-dbcb-4cea-92d6-ba3e5ceb4e25)

![image](https://github.com/user-attachments/assets/af565626-1850-4ee0-921c-d3de95d6cd9c)

```
Accuracy: 0.977
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        16
           1       0.94      1.00      0.97        16
           2       1.00      0.94      0.97        13

    accuracy                           0.98        45
   macro avg       0.98      0.98      0.98        45
weighted avg       0.98      0.98      0.98        45
```

## Project Structure
- `iris_classification.py`: The main script containing the entire machine learning pipeline.

## Next Steps
- Experiment with different algorithms like Decision Trees or Support Vector Machines (SVM).
- Tune hyperparameters to improve accuracy.
- Build a user interface using Flask or Streamlit.

## Resources
- [Iris Dataset on UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- [Scikit-learn Documentation](https://scikit-learn.org/)

## License
This project is for educational purposes. Feel free to use it and improve upon it!
