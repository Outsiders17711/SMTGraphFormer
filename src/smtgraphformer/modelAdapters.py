from .basics import *
from .contextEncoding import TransformBundle
from .dataIntegration import atbBuilder, atbDataPreprocessor
from .utils.helpers import yamlLoader

__all__ = [
    "createCanonicalBuilds",
    "defaultFeatures",
    "tfmTripLevelSMT",
    "comparePredsTargs",
    "evaluatePredictions",
]


def createCanonicalBuilds(fp: str | Path) -> "atbBuilder":
    # fp = "??/atbData-May2024-stopLevel-[fPM.eST.eLU.eDW].pkl"
    dsRanges = {
        "train": ("2024-05-01", "2024-05-21"),
        "valid": ("2024-05-22", "2024-05-26"),
        "test": ("2024-05-27", "2024-05-31"),
    }

    builder = atbBuilder(verbose=False)
    dataBCS = builder.buildCanonicalStops(fp)
    ds_splits = builder.buildSplitPlan(dataBCS, dsRanges["train"], dsRanges["valid"], dsRanges["test"])

    bundle = builder.fitTransformBundle(dataBCS, ds_splits)
    dataFTB = builder.applyBundle(dataBCS, bundle, ds_splits)
    sattrs = builder.extractStopAttributes(bundle)
    lattrs = builder.extractLineAttributes()

    return builder


def defaultFeatures(bundle: TransformBundle, category: str) -> list[str]:
    """get default input/target features from specified TransformBundle and feature category"""
    assert category in ("inputs", "targets"), "!!!"

    if category == "targets":
        return [f"tfm.{f}" for f in bundle.scTarget]

    # inputs
    cyclics = []
    for f in bundle.scCyclic:
        cyclics.extend([f"tfm.{f}.sin", f"tfm.{f}.cos"])

    return [
        "StopIdentifier",
        *[f"tfm.{f}" for f in bundle.scCat],
        *[f"tfm.{f}" for f in bundle.scCont],
        *cyclics,
    ]


def _stringify(values: pd.Series, delimiter: str) -> str:
    return delimiter.join(map(str, values.tolist()))


def tfmTripLevelSMT(tfmData: pd.DataFrame, delimiter: str = ",") -> pd.DataFrame:
    schema = yamlLoader(Path(__file__).with_name("dataSchema.yaml"))
    scs = schema["columns"]
    astc = schema["tripContext"]

    cyclics = list(scs["cyclic"].keys())
    context_map = {
        **{c: f"tfm.{c}" for c in astc["cat"]},
        **{c: f"tfm.{c}" for c in astc["cont"]},
        **{f"{c}.sin": f"tfm.{c}.sin" for c in cyclics},
        **{f"{c}.cos": f"tfm.{c}.cos" for c in cyclics},
    }
    sequence_map = {
        "StopScheduledArrival": "tfm.StopScheduledArrival",
        "StopSequence": "StopSequence",
        "StopIdentifier": "StopIdentifier",
        **{c: f"tfm.{c}" for c in scs["target"]},
    }

    required = ["dateTrip", "$split", "StopSequence", *context_map.values(), *sequence_map.values()]
    missing = [c for c in required if c not in tfmData.columns]
    assert not missing, f"missing required columns: {missing}"

    rows = []
    for dt, trip in tqdm(tfmData.groupby("dateTrip", observed=True), desc="transforming trips"):
        trip = trip.copy()
        trip["$order"] = trip["StopSequence"].astype(int)
        trip = trip.sort_values("$order").drop(columns="$order")

        flat = {"dateTrip": dt, "$split": trip["$split"].iloc[0]}
        for dst, src in context_map.items():
            flat[dst] = trip[src].iloc[0]

        for dst, src in sequence_map.items():
            flat[dst] = _stringify(trip[src], delimiter)

        rows.append(flat)

    trips = pd.DataFrame(rows).sort_values(["$split", "dateTrip"]).reset_index(drop=True)
    trips[astc["cat"]] = trips[astc["cat"]].astype(int)
    trips = atbDataPreprocessor.downcastints(trips)
    return trips


def comparePredsTargs(preds: pd.DataFrame, targs: pd.DataFrame) -> pd.DataFrame:
    assert list(preds.columns) == list(targs.columns), "!!!"
    df = pd.DataFrame(index=targs.index)
    for c in targs.columns:
        df[f"target.{c}"] = targs[c].to_numpy()
        df[f"pred.{c}"] = preds[c].to_numpy()
    return df


def evaluatePredictions(
    bundle: TransformBundle,
    preds: pd.DataFrame | np.ndarray,
    targs: pd.DataFrame | np.ndarray,
    *,
    f_target: List[str],
    raw_evaluation: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds, targs = preds.copy(), targs.copy()

    # inverse-transform predictions and targets back to raw space for evaluation
    if raw_evaluation:
        preds = bundle.inverseTargets(preds, f_target, clip=True)
        targs = bundle.inverseTargets(targs, f_target, clip=False)

    def _align(values):
        if not isinstance(values, pd.DataFrame):
            return pd.DataFrame(values, columns=f_target)

        values.columns = [c.removeprefix("tfm.") for c in values.columns]
        return values

    preds, targs = _align(preds), _align(targs)
    assert list(preds.columns) == list(targs.columns) == f_target, "!!!"

    l_metrics = []
    for c in f_target:
        # mean absolute error
        diff = preds[c] - targs[c]
        m_mae = np.abs(diff).mean()
        # root mean squared error
        squared = np.square(diff)
        m_rmse = np.sqrt(squared.mean())
        # coefficient of determination
        centred = targs[c] - targs[c].mean()
        denom = np.square(centred).sum()
        m_r2 = 1.0 - (squared.sum() / denom) if denom != 0 else 0.0

        l_metrics.append({"target": c, "rmse": float(m_rmse), "mae": float(m_mae), "r2": float(m_r2)})

    metrics = pd.DataFrame(l_metrics).round(6)
    comparison = comparePredsTargs(preds, targs).round(2)
    return metrics, comparison


class adapterSAINT: ...
