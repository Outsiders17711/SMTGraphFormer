import pickle

from .basics import *
from .contextEncoding import TransformBundle
from .utils.dataloading import *
from .utils.helpers import yamlLoader

__all__ = [
    "atbDataPreprocessor",
    "atbBuilder",
    "atbPipeline",
]


class atbDataPreprocessor:
    def __init__(self, fp: str | Path, verbose=False):
        self.data = pd.read_pickle(fp)
        self.verbose = verbose
        self.schema = yamlLoader(Path(__file__).with_name("dataSchema.yaml"))

        self.is_aligned = False
        self.check(self.data)

    def check(self, data: pd.DataFrame):
        if self.is_aligned:
            print("data is already aligned; re-instantiate the preprocessor if needed.")
            return

        required = self.schema["pklExpectedColumns"]
        missing = [c for c in required if c not in data.columns]
        assert not missing, f"data missing required columns: {missing}"
        printFeatureCardinality(data) if self.verbose else None

    @staticmethod
    def downcastints(df: pd.DataFrame) -> pd.DataFrame:
        for c in df.select_dtypes(include=["int64", "int32", "int16"]).columns:
            df[c] = pd.to_numeric(df[c], downcast="integer")
        return df

    def engineerFeatures(self) -> pd.DataFrame:
        data = self.data.copy()

        data["ST_Delay"] = (data["StopActualArrival"] - data["StopScheduledArrival"]).clip(lower=0)

        f_hour = "TripScheduledDeparture" if "TripScheduledDeparture" in data.columns else "Date"
        data["Hour"] = pd.to_datetime(data[f_hour]).dt.hour.astype("int")

        nstops = data.groupby("dateTrip", observed=True).size()
        data["nStops"] = data["dateTrip"].map(nstops).astype("int")
        data["FLAG_TripComplete"] = data["LastStopSequence"] == (data["nStops"] - 1)

        self.data = data.copy()
        return data

    def smtAlignment(self):
        assert not self.is_aligned, "data is already aligned; re-instantiate the preprocessor if needed."

        ts = ["StopScheduledArrival", "StopActualArrival"]
        if any(self.data[t].dtype == "datetime64[ns]" for t in ts):
            self.data = convertSecondsSinceMidnight(self.data, timestamps=ts, revert=True)

        data = self.engineerFeatures()

        cc = [c for c, t in data.dtypes.items() if t == "category"]
        for c in cc:
            data[c] = data[c].astype(data[c].cat.categories.dtype)

        sf = ["StopSequence", "StopIdentifier"]
        data[sf] = data[sf].astype(str)

        bc = [c for c in data.columns if c.startswith("FLAG_")]
        for c in bc:
            data[c] = data[c].astype("str")

        mapping = {
            "Boarding": "PC_Boarding",
            "Alighting": "PC_Alighting",
            "StopTime": "ST_Dwell",
        }
        data = data.rename(columns=mapping)

        for c, mapping in self.schema["featureMappings"].items():
            reverse = {v: k for k, v in mapping.items()}
            data[c] = data[c].map(reverse).astype("int")

        dc = ["index", "StopActualArrival", "TripScheduledDeparture"]
        data = data.drop(columns=dc, errors="ignore")
        data = self.downcastints(data)
        data = autoProjectCoordinates(data)

        mapping = (
            data.loc[data["BusType"] != "No Information"]
            .groupby("Line")["BusType"]
            .agg(lambda x: x.mode()[0])
        )
        data["BusType"] = data["BusType"].mask(
            data["BusType"].isin(["No Information", "Iveco - Crossway LE 4x2"]),
            data["Line"].map(mapping),
        )

        if self.verbose:
            print("\ndata alignment completed")
            printFeatureCardinality(data)

        self.data = data.copy()
        self.is_aligned = True

    def extractStopAttributes(self) -> pd.DataFrame:
        assert self.is_aligned, "data must be aligned"

        ef = self.schema["externalFactors"]
        f_landuse, f_terrain, f_transfer = ef["landuse"], ef["terrain"], ef["transfer"]
        bfs = self.schema["baseFeatures"]["stop"]
        f_attributes = ["Date", *bfs, *f_landuse, *f_terrain, *f_transfer]

        rows = []
        for _, deets in self.data.groupby("StopIdentifier", observed=True):
            deets = deets[f_attributes].copy()
            deets["Date"] = deets["Date"].dt.year  # type:ignore
            deets = deets.drop_duplicates(ignore_index=True)

            year = deets["Date"].max()
            unique = {"Date": year}
            src = ["StopIdentifier", "StopName", "Longitude", "Latitude"]
            vals = deets.loc[deets["Date"] == year, src].mode().iloc[0]
            unique.update(vals.to_dict())
            unique["TransferStop"] = deets["TransferStop"].max()
            unique["StopType"] = deets.loc[deets["TransferStop"].idxmax(), "StopType"]
            unique.update(deets[f_landuse].max().to_dict())
            vals = deets.loc[deets["Date"] == year, f_terrain].mean().round(4)  # type:ignore
            unique.update(vals.to_dict())
            rows.append(unique)

        sattrs = pd.DataFrame(rows).sort_values(["StopName", "StopIdentifier"]).reset_index(drop=True)
        sattrs["StopIdentifier"] = sattrs["StopIdentifier"].astype(self.data["StopIdentifier"].dtype)
        assert sanityCheck(sattrs), "!!!"
        assert sattrs["StopIdentifier"].is_unique, "!!!"

        self.stopAttributes = sattrs.copy()
        return sattrs

    def extractLineAttributes(self) -> pd.DataFrame:
        assert self.is_aligned, "data must be aligned"
        assert hasattr(self, "stopAttributes"), "stop attributes must be extracted first"

        ef = self.schema["externalFactors"]
        f_landuse = ef["landuse"]

        rows = []
        for (line, direction), deets in self.data.groupby(["Line", "FLAG_TripDirection"], observed=True):
            ld_trips = deets.groupby("dateTrip", observed=True)
            longest = ld_trips.size().idxmax()
            trip = ld_trips.get_group(longest).copy()
            trip["$sequence"] = trip["StopSequence"].astype(int)
            trip = trip.sort_values("$sequence").reset_index(drop=True)
            sequence = trip["StopIdentifier"].tolist()

            flat = {
                "Line": line,
                "FLAG_TripDirection": direction,
                "nStops": len(trip),
                "StopIdentifier": ",".join(map(str, sequence)),
            }

            sattrs = self.stopAttributes.set_index("StopIdentifier", drop=False).loc[sequence].copy()
            sattrs = sattrs.reset_index(drop=True)
            assert len(sattrs) == flat["nStops"], "!!!"

            head = lambda i: sattrs["StopName"].iloc[i]
            flat["TripHeadSign"] = f"{head(0)} > {head(flat['nStops'] - 1)}"
            flat["nTransferStops"] = sattrs["StopType"].sum()
            flat["nTransfers"] = sattrs["TransferStop"].sum()
            flat.update(sattrs[f_landuse].sum().to_dict())
            flat["totalDistance"] = selfRNDR(sattrs["StopDistance"].sum())
            flat["totalElevation"] = selfRNDR(sattrs["diffElevation"].sum())
            flat["slopeElevation"] = selfRNDR(sattrs["slopeElevation"].mean())
            flat["typeElevation"] = selfRNDR(sattrs["typeElevation"].mean())
            rows.append(flat)

        lattrs = pd.DataFrame(rows).sort_values(["Line", "FLAG_TripDirection"]).reset_index(drop=True)
        assert sanityCheck(lattrs), "!!!"

        self.lineAttributes = lattrs.copy()
        return lattrs


@dataclass
class atbBuilder:
    verbose: bool = False

    def __post_init__(self):
        self.schema = yamlLoader(Path(__file__).with_name("dataSchema.yaml"))
        self.scGraph = self.schema["graphAttributes"]

        scs = self.schema["columns"]
        self.scKey, self.scMeta = scs["key"], scs["meta"]
        self.scCyclic, self.scSequence = scs["cyclic"], scs["sequence"]
        self.scTarget = scs["target"]
        self.scRaw = [*scs["cat"], *scs["cont"], *scs["cyclic"], *scs["target"]]

        astc = self.schema["tripContext"]
        self.scContextCat, self.scContextCont = astc["cat"], astc["cont"]

    def loadRaw(self, fp: str | Path) -> pd.DataFrame:
        self.fp = Path(fp)
        self.preprocessor = atbDataPreprocessor(self.fp, verbose=self.verbose)
        self.raw = self.preprocessor.data.copy()
        return self.raw.copy()

    def buildCanonicalStops(self, fp: str | Path | None = None) -> pd.DataFrame:
        if fp is not None or not hasattr(self, "preprocessor"):
            self.loadRaw(fp)  # type:ignore

        self.preprocessor.smtAlignment()
        self.alignedStops = self.preprocessor.data.copy()
        assert sanityCheck(self.alignedStops), "silent data issues detected after alignment!"

        sckm = list(dict.fromkeys([*self.scKey, *self.scMeta]))  # preserve order, remove duplicates
        canonical = self.alignedStops[sckm].copy()
        for rc in self.scRaw:
            canonical[f"raw.{rc}"] = self.alignedStops[rc]

        canonical["$ordering"] = canonical["StopSequence"].astype(int)
        canonical = canonical.sort_values(["Date", "dateTrip", "$ordering"])
        canonical = canonical.drop(columns="$ordering").reset_index(drop=True)

        canonical = atbDataPreprocessor.downcastints(canonical)
        self.canonicalStops = canonical
        return canonical.copy()

    def buildSplitPlan(
        self,
        stops: pd.DataFrame,
        dr_train: tuple[str, str],
        dr_valid: tuple[str, str],
        dr_test: tuple[str, str],
    ) -> pd.DataFrame:
        """map each `dateTrip` to a `$split` (train/valid/test) based on the provided date ranges"""
        lfnr = lambda r: (pd.Timestamp(r[0]).normalize(), pd.Timestamp(r[1]).normalize())
        ranges = {"train": lfnr(dr_train), "valid": lfnr(dr_valid), "test": lfnr(dr_test)}

        trips = stops[["dateTrip", "Date"]].drop_duplicates().copy()
        trips["Date"] = pd.to_datetime(trips["Date"]).dt.normalize()
        trips["$split"] = pd.NA

        for split, (start, end) in ranges.items():
            mask = trips["Date"].between(start, end, inclusive="both")
            trips.loc[mask, "$split"] = split

        assert sanityCheck(trips), "!!!"
        ds_splits = trips[["dateTrip", "$split"]].sort_values("dateTrip").reset_index(drop=True)
        self.splitPlan = ds_splits
        return ds_splits.copy()

    def attachSplit(self, stops: pd.DataFrame, ds_splits: pd.DataFrame) -> pd.DataFrame:
        merged = stops.merge(ds_splits[["dateTrip", "$split"]], on="dateTrip", how="left")
        assert sanityCheck(merged), "!!!"
        return merged

    def extractStopAttributes(self, bundle: TransformBundle) -> pd.DataFrame:
        assert self.preprocessor.is_aligned, "data must be aligned"
        sattrs = self.preprocessor.extractStopAttributes()

        s_ordering = [t for t, i in sorted(bundle.mapStopIDs.maps.items(), key=lambda kv: kv[1])]
        sattrs = sattrs.set_index("StopIdentifier").loc[s_ordering].reset_index()
        self.stopAttributes = sattrs
        return sattrs.copy()

    def extractLineAttributes(self) -> pd.DataFrame:
        assert self.preprocessor.is_aligned, "data must be aligned"
        assert hasattr(self, "stopAttributes"), "stop attributes must be extracted first"
        lattrs = self.preprocessor.extractLineAttributes()
        self.lineAttributes = lattrs
        return lattrs.copy()

    def aggTrainMobilityPattern(
        self,
        data: pd.DataFrame,
        bundle: TransformBundle,
        *,
        target: str = "combined",
        resolution: str = "hourly",
        aggfunc: str = "sum",
    ) -> pd.DataFrame:
        """aggregate mobility patterns from the training subset; calls original aggregation function"""
        smp = self.schema["mobilityPattern"]
        assert resolution in smp["resolution"], f"valid resolutions: {smp['resolution']}"

        if "$split" not in data.columns:
            data = self.attachSplit(data, self.splitPlan)

        train = data.loc[data["$split"] == "train"].copy()
        mapping = {c: c.replace("raw.", "") for c in train.columns}
        train = train.rename(columns=mapping)

        agg = aggregateMobilityPattern(train, target=target, resolution=resolution, aggfunc=aggfunc)

        s_ordering = [t for t, i in sorted(bundle.mapStopIDs.maps.items(), key=lambda kv: kv[1])]
        agg = agg.loc[s_ordering]
        self.mobilityPatterns = agg
        return agg.copy()

    def fitTransformBundle(self, data: pd.DataFrame, ds_splits: pd.DataFrame) -> TransformBundle:
        data = self.attachSplit(data, ds_splits) if "$split" not in data.columns else data.copy()
        ds_train = data.loc[data["$split"] == "train"].copy()
        bundle = TransformBundle(data, ds_train, self.schema)
        self.transformBundle = bundle
        return bundle

    def applyBundle(
        self,
        data: pd.DataFrame,
        bundle: TransformBundle,
        ds_splits: pd.DataFrame,
    ) -> pd.DataFrame:
        transformed = bundle.transform(data, ds_splits=ds_splits)
        self.transformedStops = transformed.copy()
        return transformed

    def saveArtefacts(self, dst: str | Path, tag: str | None = None):
        dst = Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        tag = f"-{tag}" if tag is not None else ""

        self.canonicalStops.to_pickle(f"{dst}/AtB-CanonicalStops{tag}.pkl")
        self.splitPlan.to_csv(f"{dst}/AtB-SplitPlan{tag}.csv", index=False)
        self.stopAttributes.to_csv(f"{dst}/AtB-StopAttributes{tag}.csv", index=False)
        self.lineAttributes.to_csv(f"{dst}/AtB-LineAttributes{tag}.csv", index=False)
        self.transformBundle.save(f"{dst}/AtB-TransformBundle{tag}.json")
        self.transformedStops.to_pickle(f"{dst}/AtB-TransformedStops{tag}.pkl")
        with open(f"{dst}/AtB-MobilityPatterns-Train{tag}.pkl", "wb") as f:
            pickle.dump(self.mobilityPatterns, f)


@dataclass
class atbPipeline:
    builder: atbBuilder

    def runCanonical(self, fp: str | Path) -> pd.DataFrame:
        return self.builder.buildCanonicalStops(fp)

    def runSplits(
        self,
        stops: pd.DataFrame,
        dr_train: tuple[str, str],
        dr_valid: tuple[str, str],
        dr_test: tuple[str, str],
    ) -> pd.DataFrame:
        return self.builder.buildSplitPlan(stops, dr_train, dr_valid, dr_test)

    def runTransforms(
        self,
        stops: pd.DataFrame,
        ds_splits: pd.DataFrame,
    ) -> tuple[TransformBundle, pd.DataFrame]:
        bundle = self.builder.fitTransformBundle(stops, ds_splits)
        transformed = self.builder.applyBundle(stops, bundle, ds_splits)
        return bundle, transformed

    def runGraphPrep(
        self,
        stops: pd.DataFrame,
        bundle: TransformBundle,
        *,
        target: str = "combined",
        resolution: str = "hourly",
        aggfunc: str = "sum",
    ) -> pd.DataFrame:
        return self.builder.aggTrainMobilityPattern(
            stops,
            bundle,
            target=target,
            resolution=resolution,
            aggfunc=aggfunc,
        )


def autoProjectCoordinates(df, f_lon="Longitude", f_lat="Latitude"):
    lon_min, lon_max = df[f_lon].min(), df[f_lon].max()
    lat_min, lat_max = df[f_lat].min(), df[f_lat].max()
    print(f"coordinate ranges: lon=[{lon_min:.2f}, {lon_max:.2f}], lat=[{lat_min:.2f}, {lat_max:.2f}]")

    is_geographic = (
        lon_min >= -180
        and lon_max <= 180
        and lat_min >= -90
        and lat_max <= 90
        and abs(lon_max - lon_min) < 180
    )
    if not is_geographic:
        print("coordinates appear to already be projected (values outside lat/lon range)")
        return df

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[f_lon], df[f_lat]), crs="EPSG:4326")
    crs = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(crs)

    df = df.copy()
    df[f_lon], df[f_lat] = gdf.geometry.x, gdf.geometry.y
    print(f"projected {f_lon}/{f_lat} coordinates to {crs} (in meters)")
    return df


def aggregateMobilityPattern(
    data: pd.DataFrame,
    *,
    target: str,
    resolution: str,
    aggfunc: str = "sum",
) -> pd.DataFrame:
    schema = yamlLoader(Path(__file__).with_name("dataSchema.yaml"))["mobilityPattern"]
    assert target in schema["target"], f"!!! valid options: {schema['target']}"
    assert resolution in schema["resolution"], f"!!! valid options: {schema['resolution']}"
    assert aggfunc in ["sum", "mean"], "!!! valid options: ['sum', 'mean']"

    data = data.copy()
    f_target, f_resolution = "$history", "$slot"

    if target == "boarding":
        data[f_target] = data["PC_Boarding"]
    elif target == "alighting":
        data[f_target] = data["PC_Alighting"]
    elif target == "combined":
        data[f_target] = data["PC_Boarding"] + data["PC_Alighting"]

    min_date = data["Date"].min()
    max_date = data["Date"].max()

    if resolution == "hourly":
        base = (data["Date"] - min_date).dt.days * 24
        data[f_resolution] = base + data["Hour"]
        n_slots = ((max_date - min_date).days + 1) * 24
    elif resolution == "daily":
        data[f_resolution] = (data["Date"] - min_date).dt.days
        n_slots = (max_date - min_date).days + 1
    elif resolution == "weekly":
        data[f_resolution] = (data["Date"] - min_date).dt.days // 7
        n_slots = ((max_date - min_date).days // 7) + 1
    else:
        base = (data["Date"].dt.year - min_date.year) * 12  # type:ignore
        data[f_resolution] = base + (data["Date"].dt.month - min_date.month)  # type:ignore
        n_slots = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month) + 1

    agg = data.groupby(["StopIdentifier", f_resolution], observed=True)[f_target]
    agg = agg.sum() if aggfunc == "sum" else agg.mean()
    agg = agg.unstack(fill_value=np.nan)  # type:ignore
    agg = agg.reindex(columns=range(n_slots), fill_value=np.nan)

    means = agg.mean(axis=1)
    for c in agg.columns:
        agg[c] = agg[c].fillna(means).round(4)

    agg.index = agg.index.astype(str)
    assert sanityCheck(agg), "!!!"
    return agg
