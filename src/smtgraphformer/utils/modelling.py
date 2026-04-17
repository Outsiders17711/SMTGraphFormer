import random

from ..basics import *

__all__ = [
    "GPTConfig",
    "setReproducibility",
    "nextAlignedSize",
    "estimatePatienceES",
    "estimateEvalInterval",
    # ---
    "saveConfig",
    "loadConfig",
    "saveTrainingLog",
    "loadTrainingLog",
    "saveTrainingSummary",
    "plotTrainingHistory",
]


# --- utilities for configuring dataloader, model and training loop ---
@dataclass
class GPTConfig:
    # --- data parameters ---
    batch_size: int = 64  # how many independent sequences will we process in parallel
    ctx_length: int = 0  # maximum context length for predictions; [NOTE] actual value set by dataloader
    vocab_size: int = 0  # number of tokens in the dataset; [NOTE] actual value set by dataloader
    pct_valid: float = 0.1  # fraction of data to use for validation

    # --- model parameters ---
    n_layer: int = 4  # number of transformer blocks
    n_head: int = 4  # number of attention heads in each block
    embed_dim: int = 64  # embedding dimension for each token
    dropout_pct: float = 0.0  # dropout rate for regularization
    adjacency_mask: bool = False  # adjacency masking to enforce valid transitions

    # --- optimisation parameters ---
    train_epochs: int = 512  # number of training epochs (full passes over the dataset)
    train_iters: int = 2048  # number of training iterations (sampled batches from dataset)
    eval_iters: int = 64  # number of evaluation iterations (sampled batches from dataset)
    eval_interval: int = 16  # how often to evaluate the model
    learning_rate: float = 1e-3  # learning rate for the optimizer
    early_stopping: str | None = "valid"  # "train", "valid", or None to disable early stopping
    early_stopping_patience: int = 3  # number of evaluations to wait before stopping

    # --- misc parameters ---
    model_name: str = "smtg"  # name of the model for saving checkpoints
    model_dir: str = "models"  # directory to save model checkpoints
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # device to run the model on
    seed: int = 17711  # random seed for reproducibility


def setReproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None

    generator = torch.Generator()
    generator.manual_seed(seed)

    def e_seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return generator, e_seed_worker


def nextAlignedSize(n):
    if n < 16:
        # for small inputs, use next power of two (8 for n<8, 16 for n>=8)
        return 1 << (n.bit_length())  # this gives next power of two
    else:
        # for n >= 16, use smallest multiple of 16
        return ((n + 15) // 16) * 16


def estimatePatienceES(metrics: list[dict[str, float]], cfg: GPTConfig):
    """
    estimate a good patience value for early stopping based on training history;
    check for longest gap between improvements in validation loss, and add 10% safety buffer.
    """
    evals = [m["valid"] for i, m in enumerate(metrics) if i % cfg.eval_interval == 0]
    assert len(evals) > 1, "!!!"

    n_checks = len(evals) - 1
    n_improved = sum([evals[i] > evals[i + 1] for i in range(n_checks)])
    print(f"validation loss history: initial={evals[0]:.4f} -> final={evals[-1]:.4f}")
    print(f"improvement: {evals[0] - evals[-1]:.4f} -> ", end="")
    print(f"checks with improvement: {n_improved}/{n_checks}")

    n_no_improve = 0
    max_no_improve = 0
    for i in range(n_checks):
        if evals[i] > evals[i + 1]:
            max_no_improve = max(max_no_improve, n_no_improve)
            n_no_improve = 0
        else:
            n_no_improve += 1
    max_no_improve = max(max_no_improve, n_no_improve)  # include trailing no-improve streak

    buffer = max(1, int(np.ceil(max_no_improve * 0.1)))
    est_patience = max_no_improve + buffer
    print(f"max consecutive checks without improvement: {max_no_improve}")
    print(f"estimated early stopping patience ({buffer=}): {est_patience}")


def estimateEvalInterval(dls, cfg: GPTConfig, pct: int = 1) -> int:
    """estimate eval interval from training progress percentage for training"""
    assert 1 <= pct <= 50, "!!!"

    n_train = len(dls.idxs_train)
    epoch_iters = max(1, n_train // dls.batch_size)
    train_iters = cfg.train_epochs * epoch_iters
    est_interval = max(1, int(np.ceil(train_iters * (pct / 100))))

    n_evals = 1 + ((train_iters - 1) // est_interval)
    if (train_iters - 1) % est_interval != 0:
        n_evals += 1

    print(f"estimated evaluation interval (every ~{pct}% of training): {est_interval} iters")
    print(f"estimated total evaluations: {n_evals}")
    return est_interval


# --- utilities for saving and loading training artifacts ---
def saveConfig(cfg: GPTConfig, fp: str | Path, verbose=True):
    with open(fp, "w") as f:
        json.dump(cfg.__dict__, f, indent=4)
    print(f"configuration saved to ./{fp}") if verbose else None


def loadConfig(fp: str | Path, verbose=True) -> GPTConfig:
    with open(fp, "r") as f:
        cfg_dict = json.load(f)
    cfg = GPTConfig(**cfg_dict)
    print(f"configuration loaded from ./{fp}") if verbose else None
    return cfg


def saveTrainingLog(log: List[dict], fp: str | Path, verbose=True):
    with open(fp, "w") as f:
        json.dump(log, f, indent=4)
    print(f"training log saved to ./{fp}") if verbose else None


def loadTrainingLog(fp: str | Path, verbose=True) -> List[dict]:
    with open(fp, "r") as f:
        log = json.load(f)
    print(f"training log loaded from ./{fp}") if verbose else None
    return log


def saveTrainingSummary(m, cfg: GPTConfig, *, plot=True, stats=False, verbose=False):
    metrics = loadTrainingLog(f"{m.model_dir}/{m.model_tag}/training.log", verbose=False)
    df = pd.DataFrame(metrics).round(4)  # .iloc[:-1]  # drop final evaluation after training

    if cfg.early_stopping:
        best_iter = df[cfg.early_stopping].idxmin()
    else:  # no early stopping; use iteration with lowest total loss
        best_iter = (df["train"] + df["valid"]).idxmin()
    best_stats = df.iloc[best_iter].to_frame().to_dict()  # type:ignore
    str_stats = f"ES @ {cfg.early_stopping}: {best_stats}"

    if plot:
        fp = f"{m.model_dir}/{m.model_tag}/summary.png"
        df.plot(kind="line", figsize=(10, 4), grid=True)
        plt.xlim(-1, len(df) + 1)
        plt.title(f"Summary | {m.model_tag} | {str_stats}")
        plt.tight_layout()
        plt.savefig(fp, dpi=300)
        print(f"training summary saved to ./{fp}") if verbose else None
        plt.show() if verbose else plt.close()

    if stats:
        stats = df.describe().T.drop(columns=["count", "25%", "50%", "75%"]).round(4)
        stats["$final"] = df.iloc[-1]
        stats["$best"] = df.iloc[best_iter]  # type:ignore
        print(f"{best_stats=}") if verbose else None
        return stats


def plotTrainingHistory(metrics: list[dict[str, float]], fp: str | None = None):
    lm_train = [m["train"] for m in metrics]
    lm_valid = [m["valid"] for m in metrics]

    n_iters = len(lm_train) - 1
    n_improved = sum([lm_train[i] > lm_train[i + 1] for i in range(n_iters)])
    print(f"training loss history: initial={lm_train[0]:.4f} -> final={lm_train[-1]:.4f}")
    print(f"improvement: {lm_train[0] - lm_train[-1]:.4f} -> ", end="")
    print(f"iters with improvement: {n_improved}/{n_iters}")

    df = pd.DataFrame(metrics).round(4).iloc[:-1]  # drop final evaluation after training
    m_best = df.loc[df["valid"].idxmin()]
    m_summary = f"iter={m_best.name}, train={m_best['train']}, valid={m_best['valid']}"
    model_tag = Path(fp).parent.name if fp else "smt24"
    print(f"best metrics @{model_tag}: {m_summary}")

    # smooth curve with moving average
    window = 7
    lms_train = np.convolve(lm_train, np.ones(window) / window, mode="valid")

    # full training progress
    plt.figure(figsize=(10, 4), dpi=300)
    plt.plot(lm_train, alpha=0.4, label="train (raw)")
    plt.plot(range(window - 1, len(lm_train)), lms_train, linewidth=1.5, label="train (smoothed)")
    plt.plot(lm_valid, alpha=0.8, label="valid (raw)", linewidth=1.5)

    plt.xlim(-window, n_iters + window)  # limit x-axis to data range (with padding)
    plt.xlabel("iters")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f"Training History | {model_tag} | {m_summary}")

    plt.tight_layout()
    plt.savefig(fp, bbox_inches="tight") if fp else None
    plt.show()
