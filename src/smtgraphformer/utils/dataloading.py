from ..basics import *

__all__ = [
    "sanityCheck",
    "df_csver",
    "selfRNDR",
    "convertSecondsSinceMidnight",
    "printFeatureCardinality",
]


def sanityCheck(data, verbose=False):
    print("HEI: Doing sanity checks ... No issues?? ", end="") if verbose else None
    any_nans = data.isna().any().any()
    any_empties = data.isin(["", " "]).any().any()
    result = not (any_nans or any_empties)
    print(result) if verbose else None
    return result


def df_csver(df, tag: str | Path = "df.tmp"):
    Path(tag).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{tag}.csv", index=False)
    print(f"dataframe saved to ./{tag}.csv")


def selfRNDR(item, decimals=4, debug=False) -> Union[float, list, dict]:
    decimals = max(0, decimals)
    formatter = f"{{:.{decimals}f}}"
    print(f"{type(item)}->{item}") if debug else None

    if isinstance(item, (float, np.floating)):
        return float(formatter.format(item))
    if isinstance(item, list):
        return [selfRNDR(sub_item, decimals) for sub_item in item]
    if isinstance(item, dict):
        return {key: selfRNDR(value, decimals) for key, value in item.items()}
    if isinstance(item, np.ndarray):
        return selfRNDR(item.tolist(), decimals)
    if hasattr(item, "_dict"):
        return selfRNDR(dict(item), decimals)
    return item


def convertSecondsSinceMidnight(data: pd.DataFrame, timestamps: list[str], revert=False, minutes=False):
    """convert between datetime objects and seconds/minutes since midnight"""
    if revert:  # to seconds/minutes since midnight
        # subtract the Date column and convert the timedelta to seconds/minutes
        for f in timestamps:
            data[f] = (data[f] - data["Date"]).dt.total_seconds()  # type:ignore
            data[f] = data[f].div(60).round().astype(int) if minutes else data[f].astype(int)
    else:  # convert to datetime objects
        # convert seconds/minutes to timedelta, then combine with Date column
        unit = "m" if minutes else "s"
        for f in timestamps:
            data[f] = data["Date"] + pd.to_timedelta(data[f], unit=unit)

    return data


def getMemoryUsage(data):
    print(f"{data.shape=} ... ", end="")
    memory_usage = data.memory_usage(deep=False).sum() / (1024**3)
    print(f"{memory_usage:.2f}GB")


def primitiveDtypes(df: pd.DataFrame) -> dict:
    mapping = {
        "int64": "int",
        "int32": "int",
        "float64": "float",
        "float32": "float",
        "object": "str",
        "bool": "bool",
        "datetime64[ns]": "dtime",
        "datetime64[us]": "dtime",
        "datetime64[ms]": "dtime",
        "geometry": "geom",
    }
    return {c: mapping.get(str(dtype), str(dtype)) for c, dtype in df.dtypes.items()}


def printFeatureCardinality(data, ncols=3, linelength=105):
    getMemoryUsage(data)
    deets = zip(data.columns, data.nunique(), primitiveDtypes(data).values())
    ml_card = max(len(str(i)) for i in data.nunique())
    ml_name = max(len(c) for c in data.columns)

    ml_left = (linelength - (ml_card + 3) * ncols - 3 * (ncols - 1)) // ncols
    ml_left = min(ml_left, ml_name + 8)

    items = ""
    for i, (c_name, c_card, p_type) in enumerate(deets):
        if i % ncols == 0:
            items += "\n"
        else:
            items += " " * 3

        left, right = f"[{p_type}] {c_name}", f"{c_card}"
        items += f"{left[:ml_left]:<{ml_left}} : {right:<{ml_card}}"

    print(items.strip())
