# Disease Prediction Using Machine Learning

This project is a disease prediction system that utilizes machine learning techniques to predict diseases based on user-input symptoms. The system is built using Python and its essential libraries, and features a graphical user interface (GUI) for ease of use.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [GUI Usage](#gui-usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites
Make sure you have the following libraries installed before running the project:
- pandas
- numpy
- scikit-learn
- tkinter

You can install these libraries using pip:
```bash
pip install pandas numpy scikit-learn
```

## Installation
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your_username/disease-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd disease-prediction
   ```
3. Ensure all required libraries are installed.

## Dataset
The dataset used for this project is `main.csv`, which contains symptoms as columns and a `prognosis` column for the disease diagnosis. The dataset should be placed in the root directory of the project.

## Model Training
The model is trained using the Naive Bayes classifier from the scikit-learn library. The dataset is split into training and testing sets to evaluate the model's performance. 

### Training the Model
The training process involves reading the dataset, splitting it into training and testing sets, and then fitting the model:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Reading the dataset
data = pd.read_csv("main.csv")
symptoms = list(data.columns.values)
symptoms.pop()
disease = list(set(data["prognosis"]))

# Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(data[symptoms], data[["prognosis"]], random_state=104, test_size=0.25, shuffle=True)

# Training the model
model = MultinomialNB()
model.fit(x_train, np.ravel(y_train))

# Checking the accuracy of the model
y_pred = model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
```

## GUI Usage
The graphical user interface (GUI) is created using the Tkinter library. The GUI allows users to input symptoms and predict the disease based on those symptoms.

### Running the GUI
Run the `disease_prediction.py` script to start the GUI:
```bash
python disease_prediction.py
```

### Using the GUI
1. Enter symptoms into the provided dropdown menus.
2. Click the "Predict" button to display the predicted disease.

## Project Structure
```
disease-prediction/
├── main.csv
├── bg.png
├── disease_prediction.py
├── README.md
```

- `main.csv`: The dataset file containing symptoms and disease prognosis.
- `bg.png`: Background image for the GUI.
- `disease_prediction.py`: Main script containing the code for training the model and the GUI.
- `README.md`: Project documentation.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
