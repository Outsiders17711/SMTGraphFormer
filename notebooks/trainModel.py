import argparse

import torch
from smtgraphformer import *
from smtgraphformer.contextEncoding import TransformBundle, configCE
from smtgraphformer.smtGraphFormer import *

__all__ = [
    "main",
    "parseConfig",
    "locateWorkspace",
    "locateModels",
]

# ensure non-interactive matplotlib backend when script is called from notebook
if os.environ.get("MPLBACKEND", "").startswith("module://matplotlib_inline"):
    os.environ["MPLBACKEND"] = "Agg"


def parseConfig(fp: Path | str) -> tuple[SMTConfig, dict[str, Any]]:
    config = yamlLoader(fp)
    assert "model" in config, "!!!"
    assert "extra" in config, "!!!"

    # get default data types (yaml parsing can be inconsistent)
    dypes = {k: type(v) for k, v in vars(SMTConfig()).items()}
    d_cfgModel = {k: dypes[k](v) for k, v in config["model"].items()}

    cfgModel = SMTConfig(**d_cfgModel)
    cfgModel.model_dir = locateModels(d_cfgModel.get("model_dir", "../models"))
    cfgModel.model_name = "smtM24"

    cfgExtra = {
        "n_epochs": config["extra"].get("gae_train_epochs", 4096),
        "pct": config["extra"].get("smt_eval_pct_progress", 1.0),
        "raw_evaluation": config["extra"].get("smt_final_eval_raw", True),
        "teacher_forcing": config["extra"].get("smt_final_eval_forced", True),
    }
    return cfgModel, cfgExtra


def locateWorkspace():
    return Path(__file__).resolve().parents[1]


def locateModels(dir: str):
    workspace = str(locateWorkspace())

    l_location = list(Path(dir).parts)
    if l_location[0] == "..":
        l_location[0] = workspace
    else:
        l_location.insert(0, workspace)

    location = "/".join(l_location)
    while "//" in location:
        location = location.replace("//", "/")

    return location


def backupConfig(file: Path | str, dst: Path | str) -> None:
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(file, f"{dst}/trainModel.yaml")


def main(file: Path | str) -> None:
    setDisplayOptions()
    sr = setReproducibility(17711)

    cfgModel, cfgExtra = parseConfig(file)
    m_tag = f"{cfgModel.model_name}-{datetime.now().strftime('%d%m%H%M')}"
    m_savepath = Path(f"{cfgModel.model_dir}/{m_tag}")
    fp_output = m_savepath / "outputs.log"
    backupConfig(file, dst=m_savepath)
    print(f"model tag: {m_tag}; artifacts and logs will be saved to: {m_savepath}")

    with pipeCellOutput(str(fp_output)):
        # --- load saved artefacts ---
        root = f"{locateWorkspace()}/data"
        bundle = TransformBundle.load(f"{root}/AtB-TransformBundle.json")
        ds_splits = pd.read_csv(f"{root}/AtB-SplitPlan.csv")
        sattrs = pd.read_csv(f"{root}/AtB-StopAttributes.csv")
        lattrs = pd.read_csv(f"{root}/AtB-LineAttributes.csv")
        tripsSMT = pd.read_pickle(f"{root}/AtB-tfmTripLevelSMT.pkl")
        # mobilityPatterns = pklLoader(f"{root}/AtB-MobilityPatterns-Train.pkl")
        stoprms = np.load(f"{root}/AtB-buildRelationalMatrices.npy")

        # --- sanity checks ---
        assert stoprms.ndim in (3, 4), "!!!"
        assert stoprms.shape[-1] == stoprms.shape[-2], "!!!"
        assert ds_splits["$split"].isin(["train", "valid", "test"]).all(), "!!!"
        print(f"{tripsSMT.shape=}")
        print(f"{sattrs.shape=}, {lattrs.shape=}")
        print(f"{stoprms.shape=}")
        print(f"split counts={tripsSMT.groupby('$split', observed=True).size().to_dict()}")
        print(f"targets={list(bundle.targetTransforms)}")

        # --- dataloader setup ---
        dls = SMTDataloader(tripsSMT, cfg=cfgModel, bundle=bundle, sattrs=sattrs)
        cfgModel.vocab_size = dls.vocab_size
        cfgModel.ctx_length = dls.ctx_length
        cfgModel.eval_interval = estimateEvalInterval(dls, cfgModel, pct=cfgExtra["pct"])

        cfgContext = configCE(
            lc_cats=dls.ce_lccats,
            conts=dls.ce_nconts,
            output=cfgModel.trip_context_dim,  # type:ignore
        )

        print(f"{dls.info()=}")
        print(f"special tokens: {dls.stoi['<SOS>']=}, {dls.stoi['<EOS>']=}, {dls.stoi['<PAD>']=}")
        print(f"context: ce_lccats={dls.ce_lccats}, ce_nconts={dls.ce_nconts}")

        # --- graph autoencoder initialisation ---
        t_stoprms = torch.tensor(stoprms, dtype=torch.float32)
        t_stoprms = t_stoprms.unsqueeze(0) if stoprms.ndim == 3 else t_stoprms
        print(f" > stacked matrices tensor: {t_stoprms.shape}")

        # --- graph autoencoder training ---
        gaeModel, gaeEmbeddings = trainGAEModelEpochs(
            t_stoprms, cfg=cfgModel, n_epochs=cfgExtra["n_epochs"]
        )
        torch.save(gaeModel.state_dict(), f"{m_savepath}/gaeModel.pt")
        np.save(f"{m_savepath}/gaeEmbeddings.npy", gaeEmbeddings.cpu().numpy())

        lfp = lambda name: f"{name:>20}:"
        print(lfp("graph embeddings"), f"{gaeEmbeddings.shape}")
        print(lfp("vocabulary size"), f"{gaeEmbeddings.shape[0]}")
        print(lfp("embedding dimension"), f"{gaeEmbeddings.shape[1]}")

        # --- model initialisation and forward pass ---
        model = SMTGraphFormer(
            cfg=cfgModel,
            cfgContext=cfgContext,
            graph_embeddings=gaeEmbeddings,
            stop_features=dls.stopFeatures,
        )
        model.model_tag = m_tag
        model.dryrun(dls)

        # --- training ---
        log_metrics = trainSMTModelEpochs(
            m=model, dls=dls, cfg=cfgModel, save_model=True, final_eval=True
        )
        plotTrainingHistory(log_metrics, fp=f"{m_savepath}/history.png")

        # --- final evaluation ---
        l_metrics = []
        comparisons = {}
        for split in ["train", "valid", "test"]:
            s_metrics, s_comparison = smtFinalEvaluation(
                model,
                dls,
                bundle,
                split,
                raw_evaluation=cfgExtra["raw_evaluation"],
                teacher_forcing=cfgExtra["teacher_forcing"],
            )
            s_metrics.insert(0, "$split", split)
            l_metrics.append(s_metrics)
            comparisons[split] = s_comparison

        metrics = pd.concat(l_metrics, ignore_index=True)
        print(metrics.tail(4))

        # --- save evaluation metrics ---
        suffix = ".forced" if cfgExtra["teacher_forcing"] else ".autoreg"
        df_csver(metrics, tag=f"{m_savepath}/metrics{suffix}")
        # for s, comp in comparisons.items():
        #     df_csver(comp, tag=f"{m_savepath}/comparison-{s}")

    cleanLogFile(fp_output, fp_output)
    print("training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="path to training config YAML file")
    args = parser.parse_args()

    file = Path(args.config).resolve()
    assert file.exists(), "!!!"

    main(file)
