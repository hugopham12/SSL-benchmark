from enum import Enum


class RunMode(str, Enum):
    classification = "Classification"
    imputation = "Imputation"
    regression = "Regression"
    self_supervised = "self_supervised"
