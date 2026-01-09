from __future__ import annotations

import pandas as pd
import mlflow.pyfunc
from autogluon.tabular import TabularPredictor


class AutoGluonPyFuncModel(mlflow.pyfunc.PythonModel):
    """
    MLflow Model Registry에 올리기 위한 pyfunc 래퍼.
    내부적으로 AutoGluon Predictor를 로드해서 predict를 호출한다.
    """

    def load_context(self, context):
        # artifacts로 전달된 AutoGluon 모델 폴더 경로
        ag_path = context.artifacts["ag_model_dir"]
        self.predictor = TabularPredictor.load(ag_path)

    def predict(self, context, model_input):
        # model_input은 pandas DataFrame 형태로 들어오는 것이 표준
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        preds = self.predictor.predict(model_input)
        return preds
