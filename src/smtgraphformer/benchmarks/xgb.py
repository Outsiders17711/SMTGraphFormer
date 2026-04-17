"""module for benchmarking SMTGraphFormer against XGBoost"""

import xgboost as xgb

from ..basics import *
from ..contextEncoding import TransformBundle
from ..dataIntegration import atbDataPreprocessor
from ..modelAdapters import defaultFeatures, evaluatePredictions
from ..utils.modelling import setReproducibility

__all__ = [
    "XGBConfig",
    "XGBModel",
    "tfmStopLevelXGB",
    "runBaselineXGB",
]


@dataclass
class XGBConfig:
    n_estimators: int = 2048
    n_jobs: int = -1
    patience: int = 15
    # ---
    colsample_bytree: float = 0.5
    eta: float = 0.04
    gamma: float = 0.12
    max_depth: int = 16
    min_child_weight: float = 8.5
    reg_alpha: float = 0.001
    reg_lambda: float = 0.0
    subsample: float = 0.9
    # ---
    model_name: str = "xgbregressor.json"
    model_dir: str = "../models/xgb"


def tfmStopLevelXGB(
    tfmData: pd.DataFrame,
    bundle: TransformBundle,
    f_input: list[str] | None = None,
    f_target: list[str] | None = None,
) -> pd.DataFrame:
    f_input = f_input or defaultFeatures(bundle, "inputs")
    f_target = f_target or defaultFeatures(bundle, "targets")

    if "StopIdentifier" not in f_input:
        print("tfmStopLevelXGB: 'StopIdentifier' not in input features; adding it automatically.")
        f_input = ["StopIdentifier", *f_input]

    required = ["$split", *f_input, *f_target]
    missing = [f for f in required if f not in tfmData.columns]
    assert not missing, f"tfmData missing required columns: {missing}"

    data = pd.DataFrame(index=tfmData.index)
    data["$split"] = tfmData["$split"]

    for f in f_input:
        if f == "StopIdentifier":
            data[f] = bundle.mapStopIDs.transform(tfmData[f]).astype("category")
        elif f.startswith("tfm.") and f.count(".") == 1 and f.removeprefix("tfm.") in bundle.scCat:
            data[f] = tfmData[f].astype("category")
        else:
            data[f] = tfmData[f].astype(float)

    for f in f_target:
        data[f] = tfmData[f].astype(float)

    data = atbDataPreprocessor.downcastints(data)
    return data


class XGBModel:
    def __init__(
        self,
        cfg: XGBConfig,
        bundle: TransformBundle,
        f_input: list[str] | None = None,
        f_target: list[str] | None = None,
    ):
        self.cfg = cfg
        self.bundle = bundle
        self.f_input = f_input or defaultFeatures(bundle, "inputs")
        self.f_target = f_target or defaultFeatures(bundle, "targets")
        # ---
        self.is_fitted = False
        sr = setReproducibility(17711)
        self.setup()  # initialise model

    def setup(self):
        self.model = xgb.XGBRegressor(
            n_estimators=self.cfg.n_estimators,
            n_jobs=self.cfg.n_jobs,
            early_stopping_rounds=self.cfg.patience,
            # ---
            max_depth=self.cfg.max_depth,
            min_child_weight=self.cfg.min_child_weight,
            gamma=self.cfg.gamma,
            subsample=self.cfg.subsample,
            colsample_bytree=self.cfg.colsample_bytree,
            eta=self.cfg.eta,
            reg_alpha=self.cfg.reg_alpha,
            reg_lambda=self.cfg.reg_lambda,
            # ---
            objective="reg:squarederror",
            multi_strategy="one_output_per_tree",
            importance_type="gain",
            tree_method="hist",
            enable_categorical=True,
            verbosity=1,
            random_state=17711,
        )

    def fit(self, tfmData: pd.DataFrame) -> "XGBModel":
        data = tfmStopLevelXGB(tfmData, self.bundle, self.f_input, self.f_target)
        train = data.loc[data["$split"] == "train"].copy()
        valid = data.loc[data["$split"] == "valid"].copy()

        X_train, y_train = train[self.f_input], train[self.f_target]
        X_valid, y_valid = valid[self.f_input], valid[self.f_target]

        self.model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)])
        self.is_fitted = True
        return self

    def predictFrame(self, frame: pd.DataFrame) -> pd.DataFrame:
        assert self.is_fitted, "model must be fitted before prediction"

        preds = self.model.predict(frame[self.f_input])
        preds = np.asarray(preds, dtype=float)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        return pd.DataFrame(preds, columns=self.f_target, index=frame.index)

    def predict(self, tfmData: pd.DataFrame, subset: str | None = None) -> pd.DataFrame:
        assert self.is_fitted, "model must be fitted before prediction"

        data = tfmStopLevelXGB(tfmData, self.bundle, self.f_input, self.f_target)
        if subset is not None:
            data = data.loc[data["$split"] == subset].copy()

        return self.predictFrame(data)

    def evaluate(
        self, tfmData: pd.DataFrame, subset: str, raw_evaluation: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        assert subset in ("train", "valid", "test"), "!!!"
        assert self.is_fitted, "model must be fitted before evaluation"

        data = tfmStopLevelXGB(tfmData, self.bundle, self.f_input, self.f_target)
        frame = data.loc[data["$split"] == subset].copy()
        assert len(frame) > 0, "!!!"

        preds = self.predictFrame(frame)
        targs = frame[self.f_target].copy()

        f_target = [f.removeprefix("tfm.") for f in self.f_target]
        metrics, comparison = evaluatePredictions(
            self.bundle, preds, targs, f_target=f_target, raw_evaluation=raw_evaluation
        )
        metrics.insert(0, "$split", subset)
        return metrics, comparison

    def evaluateAll(
        self,
        tfmData: pd.DataFrame,
        subsets: tuple[str, ...] = ("train", "valid", "test"),
        raw_evaluation: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        l_metrics = []
        comparisons = {}

        for s in subsets:
            s_metrics, s_comparison = self.evaluate(tfmData, s, raw_evaluation)
            l_metrics.append(s_metrics)
            comparisons[s] = s_comparison

        metrics = pd.concat(l_metrics, ignore_index=True)
        return metrics, comparisons

    def save(self, verbose=True):
        fp = Path(self.cfg.model_dir) / self.cfg.model_name
        Path(fp).parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists
        self.model.save_model(fp)
        print(f"model saved to {fp}") if verbose else None

    def load(self, verbose=False):
        fp = Path(self.cfg.model_dir) / self.cfg.model_name
        self.model.load_model(fp)
        self.is_fitted = True
        print(f"model loaded from {fp}") if verbose else None


def runBaselineXGB(
    tfmData: pd.DataFrame,
    bundle: TransformBundle,
    cfg: XGBConfig,
    *,
    f_input: list[str] | None = None,
    f_target: list[str] | None = None,
    subsets: tuple[str, ...] = ("train", "valid", "test"),
    raw_evaluation: bool = True,
) -> tuple[XGBModel, pd.DataFrame, dict[str, pd.DataFrame]]:
    """runs the XGB baseline and returns the fitted model, metrics, and comparisons"""
    m = XGBModel(cfg=cfg, bundle=bundle, f_input=f_input, f_target=f_target)
    m.fit(tfmData)
    metrics, comparisons = m.evaluateAll(tfmData, subsets=subsets, raw_evaluation=raw_evaluation)
    return m, metrics, comparisons
