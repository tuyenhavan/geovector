from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class LandCoverClassifier:
    """Train the model for classifying the land cover types using common machine learning techniques."""

    def __init__(self, model_name: Optional[str] = "RF") -> None:
        """
        Initialize the LandCoverClassifier.

        Args:
            model_name (Optional[str]): Model short name like 'RF' (Random Forest) or 'SVM' (Support Vector Machine).
                Defaults to 'RF'.
        """
        self.model_name = model_name.lower()
        self.model = None
        self.scaler = StandardScaler()

    def preprocess(self, X: np.ndarray, scale: Optional[bool] = True) -> np.ndarray:
        """
        Preprocess and optionally scale data.

        Args:
            X (np.ndarray): Input feature data.
            scale (Optional[bool]): If True, scales the input features. Defaults to True.

        Returns:
            np.ndarray: Preprocessed features.
        """
        if scale:
            return self.scaler.fit_transform(X)
        return X

    def split_train_test_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[pd.Series, np.ndarray],
        test_size: Optional[float] = 0.2,
        random_state: Optional[int] = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split dataset into training and testing sets.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Feature data.
            y (Union[pd.Series, np.ndarray]): Target labels.
            test_size (Optional[float]): Fraction of data for testing. Defaults to 0.2.
            random_state (Optional[int]): Random seed for reproducibility. Defaults to 42.

        Returns:
            Tuple: Training and testing feature and label arrays (X_train, X_test, y_train, y_test).
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the specified model.

        Args:
            X_train (np.ndarray): Training feature data.
            y_train (np.ndarray): Training target labels.

        Raises:
            ValueError: If the model name is unsupported.
        """
        try:
            if self.model_name in ["rf", "random forest"]:
                self.model = RandomForestClassifier()
                self.model.fit(X_train, y_train)
            elif self.model_name in ["svm", "support vector machine"]:
                self.model = SVC(probability=True)
                self.model.fit(X_train, y_train)
            elif self.model_name in ["mlc", "GaussianNB"]:
                self.model = GaussianNB()
                self.model.fit(X_train, y_train)
            else:
                raise ValueError(
                    f"Unsupported model: {self.model_name}. Use 'RF' or 'SVM'."
                )
        except Exception as e:
            print(f"Error occurred during training: {e}")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the trained model.

        Args:
            X_test (np.ndarray): Testing feature data.
            y_test (np.ndarray): Testing target labels.

        Returns:
            dict: Evaluation results containing accuracy and classification report.

        Raises:
            ValueError: If the model has not been trained.
        """
        if not self.model:
            raise ValueError(
                "The model is not trained. Train the model before evaluating."
            )
        try:
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            return {"accuracy": accuracy, "report": report}
        except Exception as e:
            print(f"Error occurred during evaluation: {e}")
            return {"accuracy": None, "report": None}
