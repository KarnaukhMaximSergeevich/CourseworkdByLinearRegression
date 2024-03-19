import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


class PredictBitcoinCourse:
    """It's class predict bitcoin course by help database and linear regression"""

    def __init__(self):
        self.data_set = pd.read_csv("Housing.csv")
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.pred_y = None
        self.random_forest_Y_pred = None
        self.execute_function()

    def get_datasets(self):
        """In this function we get a features(x) and target(y) - for training and test"""

        X_value = self.data_set.iloc[:, [1]]
        Y_value = self.data_set.iloc[:, [0]]

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X_value, Y_value,
                                                                                test_size=0.3, random_state=100)

    def train_model(self):
        """In this function the model is trained"""

        reg_model = LinearRegression()
        reg_model.fit(self.X_train, self.Y_train)
        self.pred_y = reg_model.predict(self.X_test)
        print(f"Regression model score:\n{reg_model.score(self.X_test, self.Y_test)}")

    def show_result(self):
        """This function show visual result by matplotlib"""

        plt.scatter(self.X_test, self.Y_test, alpha=0.6)
        plt.plot(self.X_test, self.pred_y, color="red")
        plt.xlabel("The total area of the house in square feet")
        plt.ylabel("Price of house")
        plt.title("House price dataframe")
        plt.show()

    def execute_function(self):
        """It's function execute other function in this class"""

        self.get_datasets()
        self.train_model()
        self.show_result()


class_model = PredictBitcoinCourse()
