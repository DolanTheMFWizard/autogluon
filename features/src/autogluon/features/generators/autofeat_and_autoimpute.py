import logging

import pandas as pd
from autofeat import AutoFeatRegressor, AutoFeatClassifier
from autogluon.core.constants import PROBLEM_TYPES_CLASSIFICATION
from autoimpute.imputations import SingleImputer

from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


class AutoFeatAndAutoImpute(AbstractFeatureGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.auto_feat = None
        self.auto_impute = None

    def _fit_transform(self, X: pd.DataFrame, y: pd.Series, problem_type: str, **kwargs) -> (pd.DataFrame, dict):
        self.auto_impute = SingleImputer()

        if problem_type in PROBLEM_TYPES_CLASSIFICATION:
            self.auto_feat = AutoFeatClassifier(verbose=1, feateng_steps=3, units={})
        else:
            self.auto_feat = AutoFeatRegressor(verbose=1, feateng_steps=3, units={})

        try:
            X = self.auto_impute.fit_transform(X=X, y=y)
            X = self.auto_feat.fit_transform(X=X, y=y)
        except Exception as e:
            logger.log(15, 'Auto feat or Auto impute failed to fit')
            self.auto_feat = None
            self.auto_impute = None

        return X, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.auto_impute is None and self.auto_feat is None:
            return X
        else:
            X = self.auto_impute.transform(X)
            return self.auto_feat.transform(X)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()

    def _more_tags(self):
        return {'feature_interactions': False}
