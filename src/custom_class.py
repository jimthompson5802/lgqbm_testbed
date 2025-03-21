import lightgbm as lgb

import numpy as np
import copy
from typing import Any, Dict, Optional, Union


class CustomLGBMClassifier(lgb.LGBMClassifier):
    def __init__(self, *,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        objective: Optional[str] = None,
        class_weight: Optional[Union[Dict, str]] = None,
        min_split_gain: float = 0.0,
        min_child_weight: float = 1e-3,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
        n_jobs: Optional[int] = None,
        importance_type: str = "split",
        custom_param=1,
        **kwargs: Any,
    ):
        super().__init__(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample_for_bin=subsample_for_bin,
            objective=objective,
            class_weight=class_weight,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            importance_type=importance_type,
            **kwargs
        )
        self.custom_param = custom_param


    # def fit(self, X, y, **kwargs):
    #     print(f"fit with Custom param: {self.custom_param}")
    #     super().fit(X, y, **kwargs)

    def predict_proba(self, X,
        raw_score: bool = False,
        start_iteration: Optional[int] = 0,
        num_iteration: Optional[int] = None,
        pred_leaf: bool = False,
        pred_contrib: bool = False, 
        validate_features: bool = False,
        **kwargs
     ) -> np.ndarray:
        print(f"predict_proba with Custom param: {self.custom_param}")
        proba = super().predict_proba(
            X=X,
            raw_score=raw_score,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            # validate_features=validate_features,
            **kwargs,
        )   
        # Example: Normalize probabilities (this is just an example, you can customize as needed)
        proba = proba / np.sum(proba, axis=1, keepdims=True)
        return proba
    
    # def get_params(self, deep=True):    
    #     # params = super().get_params(deep=deep)
    #     # params['custom_param'] = self.custom_param
    #     return {'custom_param': self.custom_param, **super().get_params(deep=deep)}
    
    def __getstate__(self):
        # Return the state of the object for pickling
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # Restore the state of the object from the pickle
        self.__dict__.update(state)

    