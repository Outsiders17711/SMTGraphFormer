import math

import torch.nn as nn

from .basics import *

__all__ = [
    "configCE",
    "ContextEncoder",
    "ContinuousStats",
    "MapStopIDs",
    "MapSchemaCategories",
    "TransformTargets",
    "TransformBundle",
]


# helper functions for transform components
def hf_dnative(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def hf_rawname(column: str) -> str:
    return f"raw.{column}"


def hf_tfmname(column: str) -> str:
    return f"tfm.{column}"


@dataclass
class MapStopIDs:
    column: str
    maps: Dict[str, int]

    @classmethod
    def fit(cls, data: pd.DataFrame, column: str = "StopIdentifier") -> "MapStopIDs":
        values = pd.Index(data[column].astype(str)).drop_duplicates().tolist()
        maps = {value: index for index, value in enumerate(values)}
        return cls(column=column, maps=maps)

    def transform(self, raw) -> pd.Series:
        series = pd.Series(raw).astype(str)
        unknown = sorted(set(series.unique()) - set(self.maps))
        assert not unknown, f"unknown values in {self.column=}: {unknown}"
        encoded = series.map(self.maps)
        return encoded.astype(int)

    def revert(self, tfm) -> pd.Series:
        reverse = {index: token for token, index in self.maps.items()}
        series = pd.Series(tfm)
        unknown = sorted(set(series.unique()) - set(reverse))
        assert not unknown, f"unknown values in {self.column=}: {unknown}"
        return series.map(reverse).astype(str)

    def to_state(self) -> dict:
        return {"column": self.column, "maps": self.maps}

    @classmethod
    def from_state(cls, state: dict) -> "MapStopIDs":
        maps = {str(key): int(value) for key, value in state["maps"].items()}
        return cls(column=state["column"], maps=maps)


@dataclass
class MapSchemaCategories:
    maps: Dict[str, Dict]

    @classmethod
    def fit(cls, data: pd.DataFrame, schema: dict, columns: List[str]) -> "MapSchemaCategories":
        mappings = {}
        sf_mappings = schema["featureMappings"]

        for c in columns:
            if c in sf_mappings:
                forward = {hf_dnative(index): int(index) for index in sf_mappings[c]}
            else:
                source = hf_rawname(c) if hf_rawname(c) in data.columns else c
                uniques = pd.Index(data[source]).drop_duplicates().tolist()
                forward = {hf_dnative(value): index for index, value in enumerate(uniques)}
            mappings[c] = forward

        return cls(maps=mappings)

    def transform(self, raw, column: str) -> pd.Series:
        assert column in self.maps, f"unknown column: {column}"
        series = pd.Series(raw).map(hf_dnative)
        unknown = sorted(set(series.unique()) - set(self.maps[column]))
        assert not unknown, f"unknown values in {column=}: {unknown}"
        encoded = series.map(self.maps[column])
        return encoded.astype(int)

    def revert(self, tfm, column: str) -> pd.Series:
        assert column in self.maps, f"unknown column: {column}"
        reverse = {index: value for value, index in self.maps[column].items()}
        series = pd.Series(tfm)
        unknown = sorted(set(series.unique()) - set(reverse))
        assert not unknown, f"unknown values in {column=}: {unknown}"
        return series.map(reverse)

    def to_state(self) -> dict:
        state = {}
        for c, mapping in self.maps.items():
            state[c] = [{"key": key, "value": int(value)} for key, value in mapping.items()]
        return {"maps": state}

    @classmethod
    def from_state(cls, state: dict) -> "MapSchemaCategories":
        mappings = {}
        for c, mapping in state["maps"].items():
            parsed = {item["key"]: int(item["value"]) for item in mapping}
            mappings[c] = parsed
        return cls(maps=mappings)


@dataclass
class ContinuousStats:
    means: Dict[str, float]
    stds: Dict[str, float]

    @classmethod
    def fit(cls, data: pd.DataFrame, columns: List[str]) -> "ContinuousStats":
        means = {}
        stds = {}

        for c in columns:
            source = hf_rawname(c) if hf_rawname(c) in data.columns else c
            series = data[source].astype(float)
            means[c] = float(series.mean())

            std = float(series.std())
            stds[c] = 1.0 if np.isnan(std) or std == 0.0 else std

        return cls(means=means, stds=stds)

    def transform(self, raw, column: str) -> pd.Series:
        assert column in self.means, f"unknown column: {column}"
        series = pd.Series(raw).astype(float)
        return (series - self.means[column]) / self.stds[column]

    def inverse(self, tfm, column: str) -> pd.Series:
        assert column in self.means, f"unknown column: {column}"
        series = pd.Series(tfm).astype(float)
        return (series * self.stds[column]) + self.means[column]

    def to_state(self) -> dict:
        return {"means": self.means, "stds": self.stds}

    @classmethod
    def from_state(cls, state: dict) -> "ContinuousStats":
        means = {column: float(value) for column, value in state["means"].items()}
        stds = {column: float(value) for column, value in state["stds"].items()}
        return cls(means=means, stds=stds)


@dataclass
class TransformTargets:
    column: str
    mean: float
    std: float
    max_train: float

    @classmethod
    def fit(cls, series: pd.Series, column: str) -> "TransformTargets":
        raw = pd.Series(series).astype(float).clip(lower=0.0)
        logged = np.log1p(raw)
        mean = float(logged.mean())
        std = float(logged.std())
        std = 1.0 if np.isnan(std) or std == 0.0 else std
        max_train = float(raw.max())
        return cls(column=column, mean=mean, std=std, max_train=max_train)

    def transform(self, raw) -> pd.Series:
        raw = pd.Series(raw).astype(float).clip(lower=0.0)
        logged = np.log1p(raw)
        return pd.Series((logged - self.mean) / self.std, index=raw.index)

    def inverse(self, tfm, clip=True) -> pd.Series:
        tfm = pd.Series(tfm).astype(float)
        raw = np.expm1((tfm * self.std) + self.mean)
        raw = pd.Series(raw).clip(lower=0.0)
        if clip:
            raw = raw.clip(upper=self.max_train)
        return raw

    def to_state(self) -> dict:
        return {"column": self.column, "mean": self.mean, "std": self.std, "max_train": self.max_train}

    @classmethod
    def from_state(cls, state: dict) -> "TransformTargets":
        return cls(
            column=state["column"],
            mean=float(state["mean"]),
            std=float(state["std"]),
            max_train=float(state["max_train"]),
        )


class TransformBundle:
    def __init__(self, data: pd.DataFrame, ds_train: pd.DataFrame, schema: dict):
        self.schema = deepcopy(schema)
        self.scGraph = self.schema["graphAttributes"]

        scs = self.schema["columns"]
        self.scKey, self.scMeta = scs["key"], scs["meta"]
        self.scCat, self.scCont, self.scCyclic = scs["cat"], scs["cont"], scs["cyclic"]
        self.scTarget = scs["target"]
        self.scRaw = [*scs["cat"], *scs["cont"], *scs["cyclic"], *scs["target"]]
        self.buildColumnNames()

        self.mapStopIDs = MapStopIDs.fit(data, "StopIdentifier")
        self.categoryMaps = MapSchemaCategories.fit(data, schema, self.scCat)
        self.continuousStats = ContinuousStats.fit(ds_train, self.scCont)
        self.targetTransforms = {
            c: TransformTargets.fit(ds_train[hf_rawname(c)], c) for c in self.scTarget
        }

    def buildColumnNames(self):
        columns = []
        columns.extend([hf_tfmname(c) for c in self.scCat])
        columns.extend([hf_tfmname(c) for c in self.scCont])
        columns.extend([hf_tfmname(c) for c in self.scTarget])
        for c in self.scCyclic:
            columns.extend([f"{hf_tfmname(c)}.sin", f"{hf_tfmname(c)}.cos"])

        self.tfmCNames = columns
        self.rawCNames = [hf_rawname(c) for c in self.scRaw]

    def _get_series(self, data: pd.DataFrame, column: str) -> pd.Series:
        raw_column = hf_rawname(column)
        if raw_column in data.columns:
            return data[raw_column]
        return data[column]

    def transform(self, data: pd.DataFrame, ds_splits: pd.DataFrame | None = None) -> pd.DataFrame:
        data = data.copy()
        if ds_splits is not None and "$split" not in data.columns:
            data = data.merge(ds_splits[["dateTrip", "$split"]], on="dateTrip", how="left")

        tfmd = data[[*self.scKey, *self.scMeta, "$split"]].copy()
        for c in self.rawCNames:
            assert c in data.columns, "!!!"
            tfmd[c] = data[c]

        for c in self.scCat:
            tfmd[hf_tfmname(c)] = self.categoryMaps.transform(self._get_series(data, c), c)

        for c in self.scCont:
            tfmd[hf_tfmname(c)] = self.continuousStats.transform(self._get_series(data, c), c)

        for c, config in self.scCyclic.items():
            period, start = config
            values = self._get_series(data, c).astype(float) - start
            theta = 2 * math.pi * values / period
            tfmd[f"{hf_tfmname(c)}.sin"] = np.sin(theta)
            tfmd[f"{hf_tfmname(c)}.cos"] = np.cos(theta)

        for c in self.scTarget:
            tfmd[hf_tfmname(c)] = self.targetTransforms[c].transform(self._get_series(data, c))

        return tfmd

    def inverseTargets(
        self,
        data: pd.DataFrame | np.ndarray,
        columns: List[str],
        clip: bool = True,
    ) -> pd.DataFrame | np.ndarray:
        if isinstance(data, pd.DataFrame):
            raw = pd.DataFrame(index=data.index)
            for c in columns:
                source = hf_tfmname(c) if hf_tfmname(c) in data.columns else c
                raw[c] = self.targetTransforms[c].inverse(data[source], clip=clip)
            return raw

        array = np.asarray(data, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        assert array.shape[1] == len(columns), "!!!"

        raw = []
        for index, c in enumerate(columns):
            restored = self.targetTransforms[c].inverse(array[:, index], clip=clip)
            raw.append(restored.to_numpy())

        return np.stack(raw, axis=1)

    def to_state(self) -> dict:
        return {
            "schema": self.schema,
            "mapStopIDs": self.mapStopIDs.to_state(),
            "categoryMaps": self.categoryMaps.to_state(),
            "continuousStats": self.continuousStats.to_state(),
            "targetTransforms": {c: ttfm.to_state() for c, ttfm in self.targetTransforms.items()},
            "rawCNames": self.rawCNames,
            "tfmCNames": self.tfmCNames,
        }

    def save(self, fp: str | Path):
        with open(fp, "w") as f:
            json.dump(self.to_state(), f, indent=4)

    @classmethod
    def load(cls, fp: str | Path) -> "TransformBundle":
        with open(fp, "r") as f:
            state = json.load(f)

        bundle = cls.__new__(cls)
        bundle.schema = deepcopy(state["schema"])
        bundle.scGraph = bundle.schema["graphAttributes"]

        scs = bundle.schema["columns"]
        bundle.scKey, bundle.scMeta = scs["key"], scs["meta"]
        bundle.scCat, bundle.scCont, bundle.scCyclic = scs["cat"], scs["cont"], scs["cyclic"]
        bundle.scTarget = scs["target"]
        bundle.scRaw = [*scs["cat"], *scs["cont"], *scs["cyclic"], *scs["target"]]

        bundle.mapStopIDs = MapStopIDs.from_state(state["mapStopIDs"])
        bundle.categoryMaps = MapSchemaCategories.from_state(state["categoryMaps"])
        bundle.continuousStats = ContinuousStats.from_state(state["continuousStats"])
        bundle.targetTransforms = {
            c: TransformTargets.from_state(ttfm) for c, ttfm in state["targetTransforms"].items()
        }
        bundle.rawCNames = [hf_rawname(c) for c in bundle.scRaw]
        bundle.buildColumnNames()
        return bundle


@dataclass
class configCE:
    lc_cats: List[int]
    conts: int
    output: int
    embed: int = 16
    hidden: int = 64
    pct_dropout: float = 0.1


class ContextEncoder(nn.Module):
    def __init__(self, cfg: configCE):
        super().__init__()
        d_input = (len(cfg.lc_cats) * cfg.embed) + cfg.conts

        self.cat_embs = nn.ModuleList([nn.Embedding(d, cfg.embed) for d in cfg.lc_cats])
        self.norm = nn.LayerNorm(d_input)
        self.mlp = nn.Sequential(
            nn.Linear(d_input, cfg.hidden),
            nn.ReLU(),
            nn.Dropout(cfg.pct_dropout),
            nn.Linear(cfg.hidden, cfg.output),
        )

    def forward(self, cats: torch.Tensor, conts: torch.Tensor) -> torch.Tensor:
        conts = conts.float()
        embs = [emb(cats[:, i]) for i, emb in enumerate(self.cat_embs)]
        x = torch.cat(embs + [conts], dim=-1)
        x = self.norm(x)
        return self.mlp(x)
