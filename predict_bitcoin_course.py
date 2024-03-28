import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error


class PredictPriceOfHouse:
    """It's class predict price of house by help database and linear regression"""

    def __init__(self):
        self.data_set = pd.read_csv("Housing.csv")
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.pred_y = None
        self.execute_function()

    def get_datasets(self):
        """In this function we get a features(x) and target(y) - for training and test"""

        X_value = self.data_set.iloc[:, [1]]
        Y_value = self.data_set.iloc[:, [0]]

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X_value, Y_value,
                                                                                test_size=0.3, random_state=100)

    def train_singl_feature_model(self):
        """In this function the model is trained"""

        reg_model = LinearRegression()
        reg_model.fit(self.X_train, self.Y_train)
        self.pred_y = reg_model.predict(self.X_test)

        reg_score = reg_model.score(self.X_test, self.Y_test)
        W_0 = reg_model.intercept_
        W_1 = reg_model.coef_

        print(f"Regression model score:\n{np.round(reg_score, 2)}"
              f"\nIntercept(W0):\n{np.round(W_0, 2)}"
              f"\nCoef(W1):\n{np.round(W_1, 2)}")

        mape = mean_absolute_percentage_error(self.Y_test, self.pred_y)

        print(f"MAPE:\n{np.round(mape, 2)}%")

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
        self.train_singl_feature_model()
        self.show_result()


class PredictPriceOfHouseM:
    """It's class predict price of house by help database and linear regression"""

    def __init__(self):
        self.data_set = pd.read_csv("Housing.csv")
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.pred_y = None
        self.execute_function()

    def get_datasets(self):
        """In this function we get a features(x) and target(y) - for training and test"""

        X_value = self.data_set.drop(["price", "furnishingstatus"], axis=1)
        X_value = X_value.replace({"yes": 1, "no": 0})
        Y_value = self.data_set.iloc[:, [0]]

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X_value, Y_value,
                                                                                test_size=0.3, random_state=100)

    def train_model(self):
        """In this function the model is trained"""

        reg_model = LinearRegression()
        reg_model.fit(self.X_train, self.Y_train)
        self.pred_y = reg_model.predict(self.X_test)

        reg_score = reg_model.score(self.X_train, self.Y_train)
        W_0 = reg_model.intercept_
        W_1 = reg_model.coef_

        print(f"Regression model score:\n{np.round(reg_score, 2)}"
              f"\nIntercept(W0):\n{np.round(W_0, 2)}"
              f"\nCoef(W1):\n{[np.round(int, 2) for int in W_1[0]]}")

        mape = mean_absolute_percentage_error(self.Y_test, self.pred_y)

        print(f"MAPE:\n{np.round(mape, 2)}%")

    def show_result(self):
        """This function show visual result by matplotlib"""
        list_of_features = ["Area", "Bedrooms", "Bathrooms", "Stories", "Mainroad", "Guestroom", "Basement",
                            "Hotwaterheating", "Airconditioning", "Parking", "Prefarea"]
        for i in range(11):
            plt.scatter(self.X_test.iloc[:, [i]], self.Y_test, alpha=0.5)
            plt.xlabel(list_of_features[i])
            plt.ylabel("Price of house")
            plt.title("House price dataframe")
            plt.show()

        print(f"First 10 value of price (Factual data):\n{self.Y_test.head(11)}")
        print(f"First 10 value of price (Predict data):\n{self.pred_y[range(11)]}")

    def execute_function(self):
        """It's function execute other function in this class"""

        self.get_datasets()
        self.train_model()
        self.show_result()


if __name__ == "__main__":
    class_model = PredictPriceOfHouse()
    class_model2 = PredictPriceOfHouseM()
