"""
module for benchmarking SMTGraphFormer against RTDL's tabular baselines: MLP, ResNet, FT-Transformer
https://github.com/yandex-research/rtdl-revisiting-models
"""

import rtdl_revisiting_models as rtdl
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..basics import *
from ..contextEncoding import TransformBundle
from ..dataIntegration import atbDataPreprocessor
from ..modelAdapters import defaultFeatures, evaluatePredictions
from ..utils.modelling import setReproducibility

__all__ = [
    "RTDLConfig",
    "MLPConfig",
    "RNetConfig",
    "FTTConfig",
    "RTDLBase",
    "MLPModel",
    "RNetModel",
    "FTTModel",
    "tfmStopLevelRTDL",
    "runBaselineRTDL",
]


@dataclass
class RTDLConfig:
    tag: str = "RTDL"
    batch_size: int = 256
    patience: int = 16
    max_epochs: int = 1000
    lr: float = 3e-4
    w_decay: float = 1e-5  # weight decay (L2 regularisation)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_name: str = "rtdl.pt"
    model_dir: str = "../models/rtdlM24"


@dataclass
class MLPConfig(RTDLConfig):
    tag: str = "MLP"
    n_blocks: int = 2
    d_block: int = 384
    # ---
    dropout: float = 0.1
    model_name: str = "mlp.pt"


@dataclass
class RNetConfig(RTDLConfig):
    tag: str = "RNet"
    n_blocks: int = 2
    d_block: int = 192
    # ---
    d_hidden: int | None = None
    d_hidden_multiplier: float = 2.0
    dropout1: float = 0.15
    dropout2: float = 0.0
    model_name: str = "rnet.pt"


@dataclass
class FTTConfig(RTDLConfig):
    tag: str = "FTT"
    n_blocks: int = 3
    d_block: int = 192
    # ---
    attention_n_heads: int = 8
    attention_dropout: float = 0.2
    ffn_d_hidden: int | None = None
    ffn_d_hidden_multiplier: float = 4 / 3
    ffn_dropout: float = 0.1
    residual_dropout: float = 0.0
    model_name: str = "ftt.pt"


AnyConfig = MLPConfig | RNetConfig | FTTConfig


def tfmStopLevelRTDL(
    tfmData: pd.DataFrame,
    bundle: TransformBundle,
    f_input: list[str] | None = None,
    f_target: list[str] | None = None,
) -> pd.DataFrame:
    f_input = f_input or defaultFeatures(bundle, "inputs")
    f_target = f_target or defaultFeatures(bundle, "targets")

    if "StopIdentifier" not in f_input:
        print("tfmStopLevelRTDL: 'StopIdentifier' not in input features; adding it automatically.")
        f_input = ["StopIdentifier", *f_input]

    required = ["$split", *f_input, *f_target]
    missing = [f for f in required if f not in tfmData.columns]
    assert not missing, f"tfmData missing required columns: {missing}"

    data = pd.DataFrame(index=tfmData.index)
    data["$split"] = tfmData["$split"]

    for f in f_input:
        if f == "StopIdentifier":
            data[f] = bundle.mapStopIDs.transform(tfmData[f]).astype(int)
        elif isCat(f, bundle):
            data[f] = catCodes(tfmData[f], f, bundle)
        else:
            data[f] = tfmData[f].astype(float)

    for f in f_target:
        data[f] = tfmData[f].astype(float)

    return atbDataPreprocessor.downcastints(data)


def isCat(feature: str, bundle: TransformBundle) -> bool:
    return (
        feature.startswith("tfm.")
        and feature.count(".") == 1
        and feature.removeprefix("tfm.") in bundle.scCat
    )


def catCodes(raw: pd.Series, feature: str, bundle: TransformBundle) -> pd.Series:
    name = feature.removeprefix("tfm.")
    values = sorted(set(bundle.categoryMaps.maps[name].values()))
    mapping = {value: index for index, value in enumerate(values)}
    coded = raw.astype(int).map(mapping)
    assert not coded.isna().any(), "!!!"
    return coded.astype(int)


def catCardinalities(feature: str, bundle: TransformBundle) -> int:
    if feature == "StopIdentifier":
        return len(bundle.mapStopIDs.maps)

    name = feature.removeprefix("tfm.")
    return len(set(bundle.categoryMaps.maps[name].values()))


class RTDLBase:
    def __init__(
        self,
        cfg: RTDLConfig,
        bundle: TransformBundle,
        f_input: list[str] | None = None,
        f_target: list[str] | None = None,
    ):
        self.cfg = cfg
        self.bundle = bundle
        self.f_input = f_input or defaultFeatures(bundle, "inputs")
        self.f_target = f_target or defaultFeatures(bundle, "targets")

        self.f_cat = [f for f in self.f_input if f == "StopIdentifier" or isCat(f, bundle)]
        self.f_cont = [f for f in self.f_input if f not in self.f_cat]
        self.cat_cardinalities = [catCardinalities(f, bundle) for f in self.f_cat]

        self.model: Any = None
        self.optimiser: Any = None
        self.history = pd.DataFrame()

        self.device = torch.device(cfg.device)
        self.is_fitted = False
        self.sr_generator, self.sr_worker = setReproducibility(17711)
        self.setup()

    def setup(self):
        raise NotImplementedError()

    def _df2tensor(self, frame: pd.DataFrame, with_targs: bool = True):
        n_rows = len(frame)

        if self.f_cont:
            x_cont = torch.as_tensor(frame[self.f_cont].to_numpy(dtype=np.float32), dtype=torch.float32)
        else:
            x_cont = torch.empty((n_rows, 0), dtype=torch.float32)

        if self.f_cat:
            x_cat = torch.as_tensor(frame[self.f_cat].to_numpy(dtype=np.int64), dtype=torch.int64)
        else:
            x_cat = torch.empty((n_rows, 0), dtype=torch.int64)

        if not with_targs:
            return x_cont, x_cat, None

        y_targ = torch.as_tensor(frame[self.f_target].to_numpy(dtype=np.float32), dtype=torch.float32)
        return x_cont, x_cat, y_targ

    def _empty2none(self, x: torch.Tensor) -> torch.Tensor | None:
        """map input cat/cont tensors with zero features to None; otherwise move to device as usual"""
        x = x.to(self.device)
        return None if x.shape[1] == 0 else x

    def loader(self, frame: pd.DataFrame, shuffle: bool, with_targs: bool = True) -> DataLoader:
        x_cont, x_cat, y_targ = self._df2tensor(frame, with_targs=with_targs)
        if with_targs:
            dataset = TensorDataset(x_cont, x_cat, y_targ)  # type:ignore
        else:
            dataset = TensorDataset(x_cont, x_cat)
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            generator=self.sr_generator,
            worker_init_fn=self.sr_worker,
        )

    def predictBatch(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        assert not isinstance(self.cfg, FTTConfig), "FFTModel requires different `predictBatch` method"

        x_cont, x_cat = map(self._empty2none, (x_cont, x_cat))  # type:ignore
        parts = []

        if x_cont is not None:
            parts.append(x_cont)

        if x_cat is not None:
            for idx, card in enumerate(self.cat_cardinalities):
                parts.append(F.one_hot(x_cat[:, idx], num_classes=card).float())

        preds = self.model(torch.column_stack(parts))
        return preds if preds.ndim > 1 else preds.unsqueeze(-1)

    def _score(self, frame: pd.DataFrame) -> float:
        losses = []
        self.model.eval()

        with torch.no_grad():
            for x_cont, x_cat, y_targ in self.loader(frame, shuffle=False, with_targs=True):
                preds = self.predictBatch(x_cont, x_cat)
                loss = F.mse_loss(preds, y_targ.to(self.device), reduction="mean")
                losses.append(float(loss.detach().cpu()))

        return float(np.mean(losses)) if losses else float("nan")

    def fit(self, tfmData: pd.DataFrame):
        data = tfmStopLevelRTDL(tfmData, self.bundle, self.f_input, self.f_target)
        train = data.loc[data["$split"] == "train"].copy()
        valid = data.loc[data["$split"] == "valid"].copy()

        best_loss = float("inf")
        best_state = deepcopy(self.model.state_dict())
        wait = 0
        history = []

        for epoch in range(self.cfg.max_epochs):
            self.model.train()
            batch_losses = []

            for x_cont, x_cat, y_targ in self.loader(train, shuffle=True, with_targs=True):
                self.optimiser.zero_grad()
                preds = self.predictBatch(x_cont, x_cat)
                loss = F.mse_loss(preds, y_targ.to(self.device), reduction="mean")
                loss.backward()
                self.optimiser.step()
                batch_losses.append(float(loss.detach().cpu()))

            train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
            valid_loss = self._score(valid)
            history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state = deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= self.cfg.patience:
                    print(f"early stopping at epoch {epoch} with best valid_loss={best_loss:.6f}")
                    break

        self.model.load_state_dict(best_state)
        self.history = pd.DataFrame(history)
        self.is_fitted = True
        return self

    def predictFrame(self, frame: pd.DataFrame) -> pd.DataFrame:
        assert self.is_fitted, "model must be fitted before prediction"

        chunks = []
        self.model.eval()
        with torch.no_grad():
            for x_cont, x_cat in self.loader(frame, shuffle=False, with_targs=False):
                preds = self.predictBatch(x_cont, x_cat)
                chunks.append(preds.detach().cpu().numpy())

        preds = np.concatenate(chunks, axis=0)
        return pd.DataFrame(preds, columns=self.f_target, index=frame.index)

    def predict(self, tfmData: pd.DataFrame, subset: str | None = None) -> pd.DataFrame:
        assert self.is_fitted, "model must be fitted before prediction"

        data = tfmStopLevelRTDL(tfmData, self.bundle, self.f_input, self.f_target)
        if subset is not None:
            data = data.loc[data["$split"] == subset].copy()

        return self.predictFrame(data)

    def evaluate(
        self, tfmData: pd.DataFrame, subset: str, raw_evaluation: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        assert subset in ("train", "valid", "test"), "!!!"
        assert self.is_fitted, "model must be fitted before evaluation"

        data = tfmStopLevelRTDL(tfmData, self.bundle, self.f_input, self.f_target)
        frame = data.loc[data["$split"] == subset].copy()
        assert len(frame) > 0, "!!!"

        preds = self.predictFrame(frame)
        targs = frame[self.f_target].copy()
        f_target = [feature.removeprefix("tfm.") for feature in self.f_target]
        metrics, comparison = evaluatePredictions(
            self.bundle, preds, targs, f_target=f_target, raw_evaluation=raw_evaluation
        )
        metrics.insert(0, "model", self.cfg.tag)
        metrics.insert(1, "$split", subset)
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

    def save(self, verbose: bool = True):
        fp = Path(self.cfg.model_dir) / self.cfg.model_name
        fp.parent.mkdir(parents=True, exist_ok=True)
        artefacts = {
            "mstate": self.model.state_dict(),
            "history": self.history,
            "f_input": self.f_input,
            "f_target": self.f_target,
            "tag": self.cfg.tag,
        }
        torch.save(artefacts, fp)
        print(f"model artefacts saved to {fp}") if verbose else None

    def load(self, verbose: bool = False):
        fp = Path(self.cfg.model_dir) / self.cfg.model_name
        artefacts = torch.load(fp, map_location=self.device, weights_only=False)
        self.model.load_state_dict(artefacts["mstate"])
        self.history = artefacts["history"]
        self.is_fitted = True
        print(f"model artefacts loaded from {fp}") if verbose else None


class MLPModel(RTDLBase):
    cfg: MLPConfig

    def setup(self):
        d_in = len(self.f_cont) + sum(self.cat_cardinalities)
        self.model = rtdl.MLP(
            d_in=d_in,
            d_out=len(self.f_target),
            n_blocks=self.cfg.n_blocks,
            d_block=self.cfg.d_block,
            dropout=self.cfg.dropout,
        ).to(self.device)

        self.optimiser = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.w_decay
        )


class RNetModel(RTDLBase):
    cfg: RNetConfig

    def setup(self):
        d_in = len(self.f_cont) + sum(self.cat_cardinalities)
        self.model = rtdl.ResNet(
            d_in=d_in,
            d_out=len(self.f_target),
            n_blocks=self.cfg.n_blocks,
            d_block=self.cfg.d_block,
            d_hidden=self.cfg.d_hidden,
            d_hidden_multiplier=self.cfg.d_hidden_multiplier,
            dropout1=self.cfg.dropout1,
            dropout2=self.cfg.dropout2,
        ).to(self.device)

        self.optimiser = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.w_decay
        )


class FTTModel(RTDLBase):
    cfg: FTTConfig

    def setup(self):
        self.model = rtdl.FTTransformer(
            n_cont_features=len(self.f_cont),
            cat_cardinalities=self.cat_cardinalities,
            d_out=len(self.f_target),
            n_blocks=self.cfg.n_blocks,
            d_block=self.cfg.d_block,
            attention_n_heads=self.cfg.attention_n_heads,
            attention_dropout=self.cfg.attention_dropout,
            ffn_d_hidden=self.cfg.ffn_d_hidden,
            ffn_d_hidden_multiplier=self.cfg.ffn_d_hidden_multiplier,
            ffn_dropout=self.cfg.ffn_dropout,
            residual_dropout=self.cfg.residual_dropout,
        ).to(self.device)

        self.optimiser = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.w_decay
        )

    def predictBatch(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        x_cont, x_cat = map(self._empty2none, (x_cont, x_cat))  # type:ignore
        preds = self.model(x_cont, x_cat)
        return preds if preds.ndim > 1 else preds.unsqueeze(-1)


def runBaselineRTDL(
    tfmData: pd.DataFrame,
    bundle: TransformBundle,
    cfg: AnyConfig,
    *,
    f_input: list[str] | None = None,
    f_target: list[str] | None = None,
    subsets: tuple[str, ...] = ("train", "valid", "test"),
    raw_evaluation: bool = True,
) -> tuple[RTDLBase, pd.DataFrame, dict[str, pd.DataFrame]]:
    """runs the specified RTDL baseline and returns the fitted model, metrics, and comparisons"""
    assert isinstance(cfg, (MLPConfig, RNetConfig, FTTConfig)), "cfg must be one of MLP/RNet/FTT configs"
    cfg2model = {MLPConfig: MLPModel, RNetConfig: RNetModel, FTTConfig: FTTModel}

    trainer = cfg2model[type(cfg)]
    m = trainer(cfg=cfg, bundle=bundle, f_input=f_input, f_target=f_target)
    m.fit(tfmData)

    metrics, comparisons = m.evaluateAll(tfmData, subsets=subsets, raw_evaluation=raw_evaluation)
    return m, metrics, comparisons
