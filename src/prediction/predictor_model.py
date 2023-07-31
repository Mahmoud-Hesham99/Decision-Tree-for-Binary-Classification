import warnings
from typing import Optional , Union

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")

class Classifier:
    """A wrapper class for the Decision Tree binary classifier.

    This class provides a consistent interface that can be used with other
    classifier models.
    The Decision Tree binary classifier is encapsulated inside this class.
    """

    model_name = "decision_tree_binary_classifier"

    def __init__(
        self,
        criterion: Optional[str] = 'gini',
        splitter: Optional[str] = 'best',
        max_depth: Optional[int] = None,
        min_samples_split: Optional[int] = 2,
        min_samples_leaf: Optional[int] = 1,
        max_features: Optional[Union[int, float, str]] = None,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: Optional[float] = 0.0,
        class_weight: Optional[Union[dict, str]] = None,
        random_state: Optional[int] = 42,
        presort: Optional[bool] = False,
        **kwargs,
    ):
        """Construct a new Decision Tree binary classifier.

 Args:
            criterion (str, optional): The function used to measure the quality of a split.
                Can be 'gini' for the Gini impurity or 'entropy' for the information gain.
                Defaults to 'gini'.
            splitter (str, optional): The strategy used to choose the split at each node.
                Can be 'best' to choose the best split or 'random' to choose the best random split.
                Defaults to 'best'.
            max_depth (int, optional): The maximum depth of the tree.
                If set to None, nodes are expanded until all leaves are pure or until they contain
                less than `min_samples_split` samples. Defaults to None.
            min_samples_split (int, optional): The minimum number of samples required to split
                an internal node.
                If a node contains fewer samples than this value, it will not be split.
                Defaults to 2.
            min_samples_leaf (int, optional): The minimum number of samples required to be at a leaf node.
                A split will only be made if it leaves at least `min_samples_leaf` training samples
                in each of the left and right branches. Defaults to 1.
            max_features (int, float, str, or None, optional): The number of features to consider when looking
                for the best split. Can be an integer, float, string, or None.
                Defaults to None.
            max_leaf_nodes (int or None, optional): The maximum number of leaf nodes.
                If None, the number of leaf nodes is unlimited. Defaults to None.
            min_impurity_decrease (float, optional): A node will be split if this split induces
                a decrease of the impurity greater than or equal to this value.
                Defaults to 0.0.
            class_weight (dict, 'balanced', or None, optional): Weights associated with classes.
                Can be set to a dictionary of class labels as keys and corresponding weights as values,
                'balanced' to automatically adjust weights inversely proportional to class frequencies,
                or None for equal weights. Defaults to None.
            random_state (int, optional): The seed used by the random number generator.
                Defaults to None.
            presort (bool, optional): Whether to presort the data to speed up the finding of best splits during training.
                Defaults to False.
        """
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.random_state = random_state
        self.model = self.build_model()

    def build_model(self) -> DecisionTreeClassifier:
        """Build a new Decision Tree binary classifier."""
        model = DecisionTreeClassifier(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            class_weight=self.class_weight,
            random_state=self.random_state,
            
        )
        return model



    def fit(self, train_inputs: pd.DataFrame, train_targets: pd.Series) -> None:
        """Fit the Random Forest binary classifier to the training data.

        Args:
            train_inputs (pandas.DataFrame): The features of the training data.
            train_targets (pandas.Series): The labels of the training data.
        """
        self.model.fit(train_inputs, train_targets)

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        return self.model.predict(inputs)

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class probabilities.
        """
        return self.model.predict_proba(inputs)

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the Random Forest binary classifier and return the accuracy.

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The labels of the test data.
        Returns:
            float: The accuracy of the Random Forest binary classifier.
        """
        if self.model is not None:
            return self.model.score(test_inputs, test_targets)
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_file_path: str) -> None:
        """Save the Random Forest binary classifier to disk.

        Args:
            model_file_path (str): The full file path (dir + file name) to which
                to save the model.
        """
        joblib.dump(self, model_file_path)

    @classmethod
    def load(cls, model_file_path: str) -> "Classifier":
        """Load the Random Forest binary classifier from disk.

        Args:
            model_file_path (str): The path to the saved model.
        Returns:
            Classifier: A new instance of the loaded Random Forest binary classifier.
        """
        model = joblib.load(model_file_path)
        return model

    def __str__(self):
        return (
            f"Model name: {self.model_name}("
            f"n_estimators: {self.n_estimators}, "
            f"n_estimators: {self.min_samples_split}, "
            f"n_estimators: {self.min_samples_leaf})"
        )


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Classifier:
    """
    Instantiate and train the predictor model.

    Args:
        train_X (pd.DataFrame): The training data inputs.
        train_y (pd.Series): The training data labels.
        hyperparameters (dict): Hyperparameters for the classifier.

    Returns:
        'Classifier': The classifier model
    """
    classifier = Classifier(**hyperparameters)
    classifier.fit(train_inputs=train_inputs, train_targets=train_targets)
    return classifier


def predict_with_model(
    classifier: Classifier, data: pd.DataFrame, return_probs=False
) -> np.ndarray:
    """
    Predict class probabilities for the given data.

    Args:
        classifier (Classifier): The classifier model.
        data (pd.DataFrame): The input data.
        return_probs (bool): Whether to return class probabilities or labels.
            Defaults to True.

    Returns:
        np.ndarray: The predicted classes or class probabilities.
    """
    if return_probs:
        return classifier.predict_proba(data)
    return classifier.predict(data)


def save_predictor_model(model: Classifier, model_file_path: str) -> None:
    """
    Save the classifier model to disk.

    Args:
        model (Classifier): The classifier model to save.
        model_file_path (str): The full file path (dir + file name) to which to
            save the model.
    """
    model.save(model_file_path)


def load_predictor_model(model_fpath: str) -> Classifier:
    """
    Load the classifier model from disk.

    Args:
        model_fpath (str): The full path (dir + file name) to the saved model.

    Returns:
        Classifier: A new instance of the loaded classifier model.
    """
    return Classifier.load(model_fpath)


def evaluate_predictor_model(
    model: Classifier, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the classifier model and return the accuracy.

    Args:
        model (Classifier): The classifier model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the classifier model.
    """
    return model.evaluate(x_test, y_test)
