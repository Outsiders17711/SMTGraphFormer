import random
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .basics import *
from .contextEncoding import ContextEncoder, TransformBundle, configCE
from .modelAdapters import evaluatePredictions
from .utils.helpers import yamlLoader
from .utils.modelling import GPTConfig, nextAlignedSize, saveConfig, saveTrainingLog

__all__ = [
    "SMTGraphFormer",
    "SMTConfig",
    "SMTDataloader",
    "trainSMTModelEpochs",
    "trainSMTScheduledSampling",
    "trainGAEModelEpochs",
    "GraphAutoEncoder",
    "buildRelationalMatrices",
    "smtFinalEvaluation",
]


@dataclass
class SMTConfig(GPTConfig):
    learning_rate: float = 3e-4  # override default
    dropout_pct: float = 0.1  # override default
    pct_warmup: float = 0.1  # fraction of training iters for lr warmup
    max_grad_norm: float = 1.0  # gradient clipping max norm
    w_decay: float = 1e-5  # weight decay (L2 regularisation)

    n_task_encoder: int = 2  # surrogate tasks
    n_task_decoder: int = 2  # primary tasks
    n_expert: int = 3  # mmoe experts
    w_surrogate_tasks: float = 0.1  # weight applied to surrogate tasks for loss calculation
    stop_attribute_dim: int | None = None  # graph embedding dimension for stop attributes
    trip_context_dim: int | None = None  # context encoding dimension for global trip context

    # --- ablation study nodes ---
    use_graph_embeddings: bool = True  # (1) use graph embeddings from autoencoder (2) none
    use_stop_features: bool = True  # (1) use transformed stop features (2) none
    use_context_encoding: bool = True  # (1) add context encoding to input (2) none
    use_surrogate_tasks: bool = True  # (1) calculate surrogate task loss (2) none
    use_mmoe: bool = True  # (1) use MMoE layer (2) directly pass decoder output to prediction heads

    # --- scheduled sampling ---
    scheduled_sampling: bool = False  # (1) use scheduled sampling (2) always teacher-force
    ss_max_probability: float = 0.5  # max probability of replacing GT with model predictions

    def __post_init__(self):
        # align unspecified dimensions with model embedding dimension
        if self.stop_attribute_dim is None:
            self.stop_attribute_dim = self.embed_dim
        if self.trip_context_dim is None:
            self.trip_context_dim = self.embed_dim

        if not self.use_surrogate_tasks:
            self.w_surrogate_tasks = 0.0  # override to zero if surrogate tasks not used


class buildRelationalMatrices:
    """
    methods to build dense stop similarity matrices from spatial/attribute/mobility data;
    methods expect raw stop information (no normalisaton, continuous dtypes);
    returns square numpy arrays with dummy entries (zeros) for special tokens appended.
    """

    def __init__(self):
        schema = yamlLoader(Path(__file__).with_name("dataSchema.yaml"))
        self.n_dummies = len(schema["specialTokens"])

    def padding(self, matrix: np.ndarray) -> np.ndarray:
        """adds dummy rows/columns for special tokens with zero similarity to all stops"""
        n_stops = matrix.shape[0]
        padded = np.zeros((n_stops + self.n_dummies, n_stops + self.n_dummies))
        padded[self.n_dummies :, self.n_dummies :] = matrix
        return padded

    def stopDistance(self, df: pd.DataFrame, theta_method: str | None = "std") -> np.ndarray:
        """handles geographic coordinates of stop locations; assumes projected CRS"""
        # vectorised pairwise distance computation
        coords = df[["Latitude", "Longitude"]].values.copy()
        DM = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)

        if theta_method == "std":
            theta = np.std(DM[DM > 0])
        elif theta_method == "mean":
            theta = np.mean(DM[DM > 0])
        elif theta_method is None:
            theta = 1.0
        else:
            raise ValueError(f"unknown theta method specified; must be 'std', 'mean', or None")

        theta = max(theta, 1e-8)
        W = np.exp(-((DM / theta) ** 2) / 2)
        np.fill_diagonal(W, 0)
        return self.padding(W)

    def attributeSimilarity(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        """handles any static attribute (numerical) data representing stop features"""
        attrs = df[features].values.copy()
        attrs = attrs / (np.linalg.norm(attrs, axis=1, keepdims=True) + 1e-8)
        SM = attrs @ attrs.T
        SM = np.maximum(SM, 0)
        np.fill_diagonal(SM, 0)
        return self.padding(SM)

    def mobilityPattern(self, df: pd.DataFrame) -> np.ndarray:
        """handles any time series data representing historical mobility patterns or demand"""
        CM = df.T.corr().values.copy()
        CM[np.isnan(CM)] = 0
        CM = np.maximum(CM, 0)
        np.fill_diagonal(CM, 0)
        return self.padding(CM)


class GraphAutoEncoder(nn.Module):
    def __init__(self, n_nodes: int, embed_dim: int, n_matrices: int, device: str):
        super().__init__()
        self.n_nodes = n_nodes
        self.embed_dim = embed_dim
        self.n_matrices = n_matrices

        self.encoder = nn.Sequential(
            nn.Linear(n_nodes * n_matrices, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

        i_layer = lambda: nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, n_nodes),
            nn.Sigmoid(),
        )
        self.decoder = nn.ModuleList([i_layer() for _ in range(n_matrices)])

        self.apply(self._init_weights)
        self.to(device)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def encode(self, matrices: torch.Tensor) -> torch.Tensor:
        """encode each stop using its row from all matrices"""
        B, M, N, N = matrices.shape
        x = matrices.view(B, N, M * N)
        embeddings = self.encoder(x)
        return embeddings

    def decode(self, embeddings: torch.Tensor) -> list[torch.Tensor]:
        """reconstruct each matrix row from stop embeddings"""
        reconstructed = [decoder(embeddings) for decoder in self.decoder]
        return reconstructed

    def forward(self, matrices: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        embeddings = self.encode(matrices)
        reconstructed = self.decode(embeddings)
        return embeddings, reconstructed

    def estimateGAELoss(self, matrices: torch.Tensor, reconstructed: list[torch.Tensor]):
        """compute mse between original matrices and reconstructed rows"""
        loss = torch.tensor(0.0, device=matrices.device)
        for i in range(self.n_matrices):
            target = matrices[:, i, :, :]
            loss += F.mse_loss(reconstructed[i], target)
        return loss / self.n_matrices


class EncoderBlock(nn.Module):
    def __init__(self, cfg: SMTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            cfg.embed_dim, cfg.n_head, dropout=cfg.dropout_pct, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(cfg.embed_dim, 4 * cfg.embed_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout_pct),
            nn.Linear(4 * cfg.embed_dim, cfg.embed_dim),
            nn.Dropout(cfg.dropout_pct),
        )
        self.ln_1 = nn.LayerNorm(cfg.embed_dim)
        self.ln_2 = nn.LayerNorm(cfg.embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x_norm = self.ln_1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + attn_out
        x = x + self.ffn(self.ln_2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, cfg: SMTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            cfg.embed_dim, cfg.n_head, dropout=cfg.dropout_pct, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            cfg.embed_dim, cfg.n_head, dropout=cfg.dropout_pct, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(cfg.embed_dim, 4 * cfg.embed_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout_pct),
            nn.Linear(4 * cfg.embed_dim, cfg.embed_dim),
            nn.Dropout(cfg.dropout_pct),
        )
        self.ln_1 = nn.LayerNorm(cfg.embed_dim)
        self.ln_2 = nn.LayerNorm(cfg.embed_dim)
        self.ln_3 = nn.LayerNorm(cfg.embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_norm = self.ln_1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=tgt_mask)
        x = x + attn_out

        x_norm = self.ln_2(x)
        cross_out, _ = self.cross_attn(x_norm, memory, memory, key_padding_mask=memory_mask)
        x = x + cross_out

        x = x + self.ffn(self.ln_3(x))
        return x


class MMoELayer(nn.Module):
    def __init__(self, input_dim: int, n_expert: int, n_tasks: int):
        super().__init__()
        required = n_tasks + 1
        assert n_expert >= required, f"{n_expert=} must be >= {n_tasks=} + 1 ({required})"
        self.n_expert = n_expert
        self.n_tasks = n_tasks

        i_expert = lambda: nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
        )
        self.experts = nn.ModuleList([i_expert() for _ in range(n_expert)])
        self.gates = nn.ModuleList([nn.Linear(input_dim, n_expert) for _ in range(n_tasks)])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        task_outputs = []

        for gate in self.gates:
            gate_weights = F.softmax(gate(x), dim=-1)
            task_out = torch.sum(expert_outputs * gate_weights.unsqueeze(-2), dim=-1)
            task_outputs.append(task_out)

        return task_outputs


class dummyMMoELayer(nn.Module):
    """MMoE layer with identity expert outputs and uniform gate weights, for ablation study"""

    def __init__(self, n_tasks: int):
        super().__init__()
        self.n_tasks = n_tasks

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [x for _ in range(self.n_tasks)]


class SMTGraphFormer(nn.Module):
    def __init__(
        self,
        cfg: SMTConfig,
        cfgContext: configCE,
        graph_embeddings: torch.Tensor,
        stop_features: torch.Tensor,
    ):
        """
        ideally, the graph embeddings tensor should be precomputed from the graph autoencoder;
        otherwise, the raw stop attributes (normalised) can be used as substitute embeddings.
        """
        super().__init__()
        self.cfg = cfg
        self.ctx_length = cfg.ctx_length
        self.device = cfg.device

        timestamp = datetime.now().strftime("%d%m%H%M")
        self.model_dir = cfg.model_dir
        self.model_tag = f"{cfg.model_name}-{timestamp}"

        self.stop_embeddings = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.time_projection = self._build_adaptor(1, cfg.embed_dim)

        self.register_buffer("graph_embeddings", graph_embeddings)
        ge_dim = graph_embeddings.size(1)  # infer dimension since actual input varies
        self.graph_projection = self._build_adaptor(ge_dim, cfg.embed_dim)
        if cfg.use_graph_embeddings:
            self.graph_gate = nn.Parameter(torch.ones(1))  # trainable gate
        else:
            self.register_buffer("graph_gate", torch.zeros(1))  # ablation: disable graph embeddings

        self.register_buffer("stop_features", stop_features)
        sf_dim = stop_features.size(1)  # infer dimension since actual input varies
        self.feature_projection = self._build_adaptor(sf_dim, cfg.embed_dim)
        if cfg.use_stop_features:
            self.feature_gate = nn.Parameter(torch.ones(1))  # trainable gate
        else:
            self.register_buffer("feature_gate", torch.zeros(1))  # ablation: disable stop features

        self.ctx_encoder = ContextEncoder(cfgContext)
        if cfg.use_context_encoding:
            self.register_buffer("context_gate", torch.ones(1))  # fixed gate
        else:
            self.register_buffer("context_gate", torch.zeros(1))  # ablation: disable context encoding

        self.encoder_layers = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layer)])
        self.encoder_ln = nn.LayerNorm(cfg.embed_dim)
        self.enc_position_embeddings = nn.Embedding(cfg.ctx_length, cfg.embed_dim)
        self.surrogate_head = self._build_adaptor(cfg.embed_dim, cfg.n_task_encoder)

        self.decoder_layers = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layer)])
        self.decoder_ln = nn.LayerNorm(cfg.embed_dim)
        self.dec_position_embeddings = nn.Embedding(cfg.ctx_length, cfg.embed_dim)
        self.input_projection = self._build_adaptor(cfg.n_task_decoder, cfg.embed_dim)

        if cfg.use_mmoe:
            self.mmoe = MMoELayer(cfg.embed_dim, cfg.n_expert, n_tasks=cfg.n_task_decoder)
        else:
            self.mmoe = dummyMMoELayer(n_tasks=cfg.n_task_decoder)  # ablation: disable MMoE layer
        self.boarding_head = self._build_adaptor(cfg.embed_dim, 1)
        self.alighting_head = self._build_adaptor(cfg.embed_dim, 1)

        self.apply(self._init_weights)
        self.to(self.device)

    def _build_adaptor(self, input_dim: int, output_dim: int) -> nn.Sequential:
        adaptor = nn.Sequential(
            nn.Linear(input_dim, self.cfg.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(self.cfg.embed_dim // 2, output_dim),
        )
        return adaptor

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def create_causal_mask(self, size: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size, device=self.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def encode(
        self,
        sids: torch.Tensor,
        ssarrival: torch.Tensor,
        trip_context: tuple[torch.Tensor, torch.Tensor],
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T = sids.shape

        x = self.stop_embeddings(sids)
        pos = self.enc_position_embeddings(torch.arange(T, device=self.device))
        x = x + pos

        time_encoding = self.time_projection(ssarrival.unsqueeze(-1))
        x = x + time_encoding

        ge = self.graph_embeddings[sids]  # type:ignore
        x = x + self.graph_gate * self.graph_projection(ge)

        sf = self.stop_features[sids]  # type:ignore
        x = x + self.feature_gate * self.feature_projection(sf)

        gtc_cats, gtc_conts = trip_context
        gtc_encoded = self.ctx_encoder(gtc_cats, gtc_conts)
        x = x + self.context_gate * gtc_encoded.unsqueeze(1)

        for layer in self.encoder_layers:
            x = layer(x, padding_mask)

        x = self.encoder_ln(x)
        return x

    def decode(
        self,
        d_sids: torch.Tensor,
        d_ssarrival: torch.Tensor,
        pc_history: torch.Tensor,
        stop_context: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """decoder input shape: (B, T, n_task_decoder)"""
        x = self.input_projection(pc_history)
        T = pc_history.size(1)

        # shared stop/time/graph representations; decoder-specific position embeddings
        x = x + self.stop_embeddings(d_sids)
        x = x + self.dec_position_embeddings(torch.arange(T, device=self.device))
        x = x + self.time_projection(d_ssarrival.unsqueeze(-1))

        ge = self.graph_embeddings[d_sids]  # type:ignore
        x = x + self.graph_gate * self.graph_projection(ge)

        sf = self.stop_features[d_sids]  # type:ignore
        x = x + self.feature_gate * self.feature_projection(sf)

        tgt_mask = self.create_causal_mask(T)
        for layer in self.decoder_layers:
            x = layer(x, stop_context, tgt_mask, padding_mask)

        x = self.decoder_ln(x)
        return x

    def forward(
        self,
        stops_info: tuple[torch.Tensor, torch.Tensor],
        trip_context: tuple[torch.Tensor, torch.Tensor],
        dec_inputs: tuple[torch.Tensor, torch.Tensor],
        padding_mask: torch.Tensor | None = None,
        enc_targets: tuple[torch.Tensor, torch.Tensor] | None = None,
        dec_targets: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        sids, ssarrival = stops_info
        if padding_mask is None:
            padding_mask = torch.zeros(sids.shape, dtype=torch.bool, device=self.device)

        stop_context = self.encode(sids, ssarrival, trip_context, padding_mask)
        surrogate_preds = self.surrogate_head(stop_context)

        # decoder receives shifted stop-level context (aligned to prediction targets)
        d_sids = sids[:, 1:]  # stop ids for positions being predicted
        d_ssarrival = ssarrival[:, 1:]  # scheduled arrival for positions being predicted

        boarding_input, alighting_input = dec_inputs
        pc_history = torch.stack([boarding_input, alighting_input], dim=-1)
        decoder_output = self.decode(d_sids, d_ssarrival, pc_history, stop_context, padding_mask)

        mmoe_outputs = self.mmoe(decoder_output)
        boarding_preds = self.boarding_head(mmoe_outputs[0]).squeeze(-1)
        alighting_preds = self.alighting_head(mmoe_outputs[1]).squeeze(-1)

        loss = None
        if dec_targets is not None:
            # attention uses padding_mask directly (true = ignore padding positions)
            # loss requires negated mask (true = select non-padding positions)
            mask = ~padding_mask

            # decoder loss: autoregressive, so slice mask to match shifted predictions
            boarding_target, alighting_target = dec_targets
            dec_mask = mask[:, 1:]  # slice to match decoder output length (excludes sot)
            primary_loss = F.mse_loss(boarding_preds[dec_mask], boarding_target[dec_mask])
            primary_loss += F.mse_loss(alighting_preds[dec_mask], alighting_target[dec_mask])

            # encoder surrogate loss: not autoregressive, use full mask
            surrogate_loss = 0.0
            if enc_targets is not None:
                delay_target, dwell_target = enc_targets
                surrogate_target = torch.stack([delay_target, dwell_target], dim=-1)
                enc_mask = mask.unsqueeze(-1).expand_as(surrogate_target)
                surrogate_loss = F.mse_loss(surrogate_preds[enc_mask], surrogate_target[enc_mask])

            loss = primary_loss + self.cfg.w_surrogate_tasks * surrogate_loss

        return {
            "boarding": boarding_preds,
            "alighting": alighting_preds,
            "delay": surrogate_preds[..., 0],
            "dwell": surrogate_preds[..., 1],
            "loss": loss,
        }

    def generate(
        self,
        stops_info: tuple[torch.Tensor, torch.Tensor],
        trip_context: tuple[torch.Tensor, torch.Tensor],
        dec_inputs: tuple[torch.Tensor, torch.Tensor] | None = None,
        padding_mask: torch.Tensor | None = None,
    ):
        """
        generate boarding/alighting predictions autoregressively, one stop at a time;
        the stops info MUST include id/arrival values for the SOS/EOS tokens.
        """
        sids, ssarrival = stops_info
        self.eval()
        if padding_mask is None:
            padding_mask = torch.zeros(sids.shape, dtype=torch.bool, device=self.device)

        if dec_inputs is None:  # init dummy passenger counts for sos tokens
            B = sids.size(0)
            boarding_input = torch.zeros((B, 1), dtype=torch.float, device=self.device)
            alighting_input = torch.zeros((B, 1), dtype=torch.float, device=self.device)
        else:
            boarding_input, alighting_input = dec_inputs

        n_tokens = min(sids.size(1) - 1, self.ctx_length - 1)  # decoder has T-1 positions
        stop_context = self.encode(sids, ssarrival, trip_context, padding_mask)
        surrogate_preds = self.surrogate_head(stop_context)

        # decoder stop-level context: use positions 1..T (target-aligned)
        d_sids = sids[:, 1:]  # (B, T-1)
        d_ssarrival = ssarrival[:, 1:]  # (B, T-1)

        for step in range(n_tokens):
            # slice context up to current generation step
            i_d_sids = d_sids[:, : step + 1]
            i_d_ssarrival = d_ssarrival[:, : step + 1]

            pc_history = torch.stack([boarding_input, alighting_input], dim=-1)
            decoder_output = self.decode(i_d_sids, i_d_ssarrival, pc_history, stop_context, padding_mask)

            mmoe_outputs = self.mmoe(decoder_output)
            boarding_preds = self.boarding_head(mmoe_outputs[0]).squeeze(-1)
            alighting_preds = self.alighting_head(mmoe_outputs[1]).squeeze(-1)

            boarding_input = torch.cat([boarding_input, boarding_preds[:, -1:]], dim=1)
            alighting_input = torch.cat([alighting_input, alighting_preds[:, -1:]], dim=1)

        return {
            "boarding": boarding_input,
            "alighting": alighting_input,
            "delay": surrogate_preds[..., 0],
            "dwell": surrogate_preds[..., 1],
        }

    def info(self) -> str:
        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        parts = [
            f"n_layers={len(self.encoder_layers)}",
            f"n_heads={self.cfg.n_head}",
            f"embed_dim={self.stop_embeddings.embedding_dim}",
            f"ctx_length={self.ctx_length}",
            f"vocab_size={self.stop_embeddings.num_embeddings}",
            f"graph_dim={self.graph_embeddings.size(1)}",  # type:ignore
            f"feature_dim={self.stop_features.size(1)}",  # type:ignore
            f"n_params={n_params:.4f}M",
            f"device={next(self.parameters()).device}",
            f"model_dir={self.model_dir}",
            f"model_tag={self.model_tag}",
        ]
        return f"{self.__class__.__name__}({', '.join(parts)})"

    @torch.no_grad()
    def dryrun(self, dls):
        """perform a dry run of the model to ensure everything is working correctly"""
        batch = dls.get_batch("train")
        outputs = self(*batch)

        shapes = ""
        for i, task in enumerate(["boarding", "alighting", "delay", "dwell"]):
            shapes += f"{task}={tuple(outputs[task].shape)}"
            shapes += ", " if i < 3 else ""

        loss = outputs["loss"]
        loss = f"{loss.item():.4f}" if loss is not None else "None"

        print(f"{self.info()}\n{dls.info()}\nshapes: {shapes} > loss={loss}")

    def save(self, dir: str | None = None, verbose=True):
        dir = dir or self.model_dir
        fp = Path(dir) / f"{self.model_tag}/model.pth"
        fp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), fp)
        print(f"model saved to ./{fp}") if verbose else None

    def load(self, tag: str, dir: str | None = None, verbose=True):
        dir = dir or self.model_dir
        self.model_dir, self.model_tag = dir, tag
        fp = Path(dir) / f"{self.model_tag}/model.pth"
        self.load_state_dict(torch.load(fp, map_location=self.device))
        print(f"model loaded from ./{fp}") if verbose else None


class SMTDataloader:
    def __init__(
        self,
        trips: pd.DataFrame,
        cfg: SMTConfig,
        bundle: TransformBundle,
        sattrs: pd.DataFrame,
    ):
        self.data = trips.copy().reset_index(drop=True)
        self.bundle = bundle
        self.mapStopIDs = bundle.mapStopIDs.maps

        self.batch_size = cfg.batch_size
        self.device = cfg.device

        self.batch_generator = torch.Generator().manual_seed(cfg.seed + 2)
        self.setup()
        self.stopFeatures = self.buildStopFeatures(sattrs)

    def setup(self):
        schema = yamlLoader(Path(__file__).with_name("dataSchema.yaml"))
        stokens = ["<SOS>", "<EOS>", "<PAD>"]
        asst = schema["specialTokens"]
        assert all(t in asst for t in stokens), "schema missing required special tokens"
        assert len(set(asst)) == len(asst), "unrecognised special tokens in schema"
        self.SOS, self.EOS, self.PAD = stokens

        # extract feature groups from schema
        scs = schema["columns"]
        astc = schema["tripContext"]
        f_cat, f_cont, f_sequence, f_target = astc["cat"], astc["cont"], scs["sequence"], scs["target"]
        f_cyclic = [f"{f}.{sc}" for f in scs["cyclic"] for sc in ("sin", "cos")]

        # validate data columns against required features
        required = ["dateTrip", "$split", *f_cat, *f_cont, *f_cyclic, *f_sequence, *f_target]
        missing = [c for c in required if c not in self.data.columns]
        assert not missing, f"trips data missing required columns: {missing}"

        # extract training/validation/test indices from data splits
        self.idxs_train = self.splits("train")
        self.idxs_valid = self.splits("valid")
        self.idxs_test = self.splits("test")

        # build vocabulary from stop identifiers; append special tokens
        tokens = [t for t, i in sorted(self.mapStopIDs.items(), key=lambda kv: kv[1])]
        self.tokens = [str(t) for t in tokens] + [self.SOS, self.EOS, self.PAD]
        self.vocab_size = len(self.tokens)

        # determine context length based on longest sequence in data
        sequences = self.data["StopIdentifier"].apply(lambda x: str(x).split(","))
        max_len = sequences.apply(len).max().item()
        self.ctx_length = nextAlignedSize(max_len + 2)

        # build token-index mappings and encoding/decoding lambdas
        self.stoi = {token: i for i, token in enumerate(self.tokens)}
        self.itos = {i: token for i, token in enumerate(self.tokens)}
        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda l: [self.itos[i] for i in l]

        # ensure dls token ordering matches data preprocessor
        for token, index in self.mapStopIDs.items():
            assert self.stoi[str(token)] == index, f"dataloader token-index mismatch for stop {token}"

        # compute fill values for continuous features: what raw 0.0 maps to in transformed space
        self.fill_values = {}
        for c, ttfm in self.bundle.targetTransforms.items():
            self.fill_values[c] = float(ttfm.transform(pd.Series([0.0])).iloc[0])
        self.fill_values["StopScheduledArrival"] = 0.0  # not target-transformed; raw zero stays zero

        self.preprocess(f_cat, f_cont, f_cyclic)

    def splits(self, subset: str) -> torch.Tensor:
        assert subset in ("train", "valid", "test"), "!!!"
        idxs = self.data.index[self.data["$split"] == subset].tolist()
        return torch.tensor(idxs, dtype=torch.long)

    def preprocess(self, f_cat: List[str], f_cont: List[str], f_cyclic: List[str]):
        # generate config info for context encoder
        self.ce_lccats = [int(self.data[f].max()) + 1 for f in f_cat]
        self.ce_nconts = len(f_cont) + len(f_cyclic)

        ds_sids = self.data["StopIdentifier"].map(self._prep_cat)
        pfpc = lambda c: partial(self._prep_cont, fill=self.fill_values[c])
        ds_ssarrival = self.data["StopScheduledArrival"].map(pfpc("StopScheduledArrival"))
        ds_boarding = self.data["PC_Boarding"].map(pfpc("PC_Boarding"))
        ds_alighting = self.data["PC_Alighting"].map(pfpc("PC_Alighting"))
        ds_delay = self.data["ST_Delay"].map(pfpc("ST_Delay"))
        ds_dwell = self.data["ST_Dwell"].map(pfpc("ST_Dwell"))

        # stop-level sequences as tensors
        self.sl_sids = torch.tensor(ds_sids.to_list(), dtype=torch.long)
        self.sl_ssarrival = torch.tensor(ds_ssarrival.to_list(), dtype=torch.float)
        self.sl_boarding = torch.tensor(ds_boarding.to_list(), dtype=torch.float)
        self.sl_alighting = torch.tensor(ds_alighting.to_list(), dtype=torch.float)
        self.sl_delay = torch.tensor(ds_delay.to_list(), dtype=torch.float)
        self.sl_dwell = torch.tensor(ds_dwell.to_list(), dtype=torch.float)

        self.gtc_cats = torch.tensor(self.data[f_cat].values, dtype=torch.long)
        self.gtc_conts = torch.tensor(self.data[f_cont + f_cyclic].values, dtype=torch.float)

    def buildStopFeatures(self, sattrs: pd.DataFrame) -> torch.Tensor:
        features = list(self.bundle.scGraph)
        stats = self.bundle.continuousStats
        assert all(f in sattrs.columns for f in features), "!!!"
        sf = torch.zeros(self.vocab_size, len(features))

        for _, row in sattrs.iterrows():
            sid = str(int(row["StopIdentifier"]))
            if sid not in self.stoi:
                continue

            idx = self.stoi[sid]
            for j, f in enumerate(features):
                value = float(row[f])
                if f in stats.means:
                    value = (value - stats.means[f]) / (stats.stds[f] + 1e-8)
                sf[idx, j] = value

        return sf

    def _prep_cat(self, seq_str: str):
        """preprocess categorical sequence with sos/eos/pad tokens"""
        stops = seq_str.split(",")
        seq = [self.SOS] + stops + [self.EOS]
        encoded = self.encode(seq)
        if len(encoded) < self.ctx_length:
            encoded = encoded + [self.stoi[self.PAD]] * (self.ctx_length - len(encoded))
        else:
            encoded = encoded[: self.ctx_length]
        return encoded

    def _prep_cont(self, seq_str: str, fill=0.0):
        """preprocess continuous sequence; uses transformed zero fill for eos/sos padding"""
        values = [float(v) for v in seq_str.split(",")]
        values = [fill] + values + [fill]
        if len(values) < self.ctx_length:
            values = values + [fill] * (self.ctx_length - len(values))
        else:
            values = values[: self.ctx_length]
        return values

    def get_batch(self, subset: str):
        assert subset in ("train", "valid", "test"), "!!!"

        # select appropriate indices for train/valid split
        idxs_subset = getattr(self, f"idxs_{subset}")
        assert len(idxs_subset) > 0, "!!!"

        # randomly sample batch indices from the split
        b_idxs = torch.randint(0, len(idxs_subset), (self.batch_size,), generator=self.batch_generator)
        b_idxs = idxs_subset[b_idxs]

        # slice pre-encoded tensors
        sids = self.sl_sids[b_idxs].to(self.device)
        ssarrival = self.sl_ssarrival[b_idxs].to(self.device)
        boarding_full = self.sl_boarding[b_idxs].to(self.device)
        alighting_full = self.sl_alighting[b_idxs].to(self.device)
        delay_targets = self.sl_delay[b_idxs].to(self.device)
        dwell_targets = self.sl_dwell[b_idxs].to(self.device)

        # slice boarding/alighting for autoregressive decoding
        boarding_inputs = boarding_full[:, :-1]
        alighting_inputs = alighting_full[:, :-1]
        boarding_targets = boarding_full[:, 1:]
        alighting_targets = alighting_full[:, 1:]

        # create padding mask
        padding_mask = sids == self.stoi[self.PAD]

        # slice pre-encoded trip context
        gtc_cats = self.gtc_cats[b_idxs].to(self.device)
        gtc_conts = self.gtc_conts[b_idxs].to(self.device)

        return (
            (sids, ssarrival),
            (gtc_cats, gtc_conts),
            (boarding_inputs, alighting_inputs),
            padding_mask,
            (delay_targets, dwell_targets),
            (boarding_targets, alighting_targets),
        )

    def info(self) -> str:
        parts = [
            f"n_trips={len(self.data)}",
            f"vocab_size={self.vocab_size}",
            f"ctx_length={self.ctx_length}",
            f"batch_size={self.batch_size}",
            f"n_train={len(self.idxs_train)}",
            f"n_valid={len(self.idxs_valid)}",
            f"n_test={len(self.idxs_test)}",
            f"device={self.device}",
        ]
        return f"{self.__class__.__name__}({', '.join(parts)})"


@torch.no_grad()
def estimateSMTLoss(m, dls, n_iters: int):
    out = {}
    m.eval()

    for split in ["train", "valid"]:
        losses = torch.zeros(n_iters)
        for i in range(n_iters):
            batch = dls.get_batch(split)
            stops_info, trip_context, dec_inputs, padding_mask, enc_targets, dec_targets = batch
            result = m(stops_info, trip_context, dec_inputs, padding_mask, enc_targets, dec_targets)
            if result["loss"] is not None:
                losses[i] = result["loss"].item()
        out[split] = losses.mean().item()

    m.train()
    return out


def lossEarlyStopping(metrics, m, cfg: GPTConfig) -> bool:
    subset = cfg.early_stopping
    eval_interval = cfg.eval_interval

    if subset == "valid":
        evals = [m[subset] for i, m in enumerate(metrics) if i % eval_interval == 0]
        patience = cfg.early_stopping_patience
    elif subset == "train":
        evals = [m[subset] for m in metrics]
        patience = cfg.early_stopping_patience * eval_interval * 2
    else:
        return False

    if len(evals) == 1:
        m.save(verbose=False)
        return False

    if len(evals) < patience + 1:
        if evals[-1] <= min(evals[:-1]):
            m.save(verbose=False)
        return False

    recent_losses = evals[-(patience + 1) :]
    best_recent_loss = min(recent_losses[:-1])
    current_loss = recent_losses[-1]

    if current_loss <= best_recent_loss:
        m.save(verbose=True)
        return False
    if recent_losses.index(best_recent_loss) == 0:
        print("early stopping criterion met; best ", end="")
        m.load(m.model_tag, verbose=True)
        return True
    return False


def trainSMTModelEpochs(m, dls, cfg: SMTConfig, save_model=True, final_eval=False):
    print("--- starting training setup ---")
    print(dls.info(), m.info(), sep="\n")
    optimiser = torch.optim.AdamW(m.parameters(), lr=cfg.learning_rate, weight_decay=cfg.w_decay)

    n_train = len(dls.idxs_train)
    n_valid = len(dls.idxs_valid)
    print(f"{n_train=}, {n_valid=}")

    epoch_iters = max(1, n_train // dls.batch_size)
    train_iters = cfg.train_epochs * epoch_iters
    eval_iters = max(1, n_valid // dls.batch_size)
    print(f"{train_iters=}, {eval_iters=} | n_epochs={cfg.train_epochs}, {epoch_iters=}")

    # lr scheduling: linear warmup then cosine decay
    warmup_iters = max(1, int(cfg.pct_warmup * train_iters))
    warmup = LinearLR(optimiser, start_factor=0.01, total_iters=warmup_iters)
    decay = CosineAnnealingLR(optimiser, T_max=train_iters - warmup_iters, eta_min=1e-6)
    scheduler = SequentialLR(optimiser, [warmup, decay], milestones=[warmup_iters])

    print("\n--- training model ---")
    m.train()
    idx_epoch = 0
    idx_epoch_batch = 0

    log_metrics = []
    with tqdm(range(train_iters), desc="training") as pbar:
        for iter in pbar:
            if idx_epoch_batch >= epoch_iters:
                idx_epoch += 1
                idx_epoch_batch = 0

            batch = dls.get_batch("train")
            stops_info, trip_context, dec_inputs, padding_mask, enc_targets, dec_targets = batch
            result = m(stops_info, trip_context, dec_inputs, padding_mask, enc_targets, dec_targets)
            loss = result["loss"] if result["loss"] is not None else torch.tensor(0.0)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=cfg.max_grad_norm)
            optimiser.step()
            scheduler.step()
            idx_epoch_batch += 1

            if iter % cfg.eval_interval == 0 or iter == train_iters - 1:
                losses = estimateSMTLoss(m, dls, n_iters=eval_iters)
                log_metrics.append(losses)
                if (cfg.early_stopping == "valid") and lossEarlyStopping(log_metrics, m, cfg):
                    break
            elif log_metrics:
                losses = {"train": loss.item(), "valid": log_metrics[-1]["valid"]}
                log_metrics.append(losses)
                if (cfg.early_stopping == "train") and lossEarlyStopping(log_metrics, m, cfg):
                    break

            pf = {
                "epoch": f"{idx_epoch + 1}/{cfg.train_epochs}",
                "train_loss": f"{losses['train']:.4f}",
                "valid_loss": f"{losses['valid']:.4f}",
            }
            pbar.set_postfix(pf)

    if cfg.early_stopping:
        m.load(m.model_tag, verbose=False)

    if final_eval:
        losses = estimateSMTLoss(m, dls, n_iters=eval_iters)
        log_metrics.append(losses)
        print(f"final evaluation: train loss {losses['train']:.4f} | valid loss {losses['valid']:.4f}")

    if save_model:
        m.save(verbose=True) if not cfg.early_stopping else None  # avoid overwriting best model
        saveConfig(cfg, fp=f"{m.model_dir}/{m.model_tag}/config.json")
        saveTrainingLog(log_metrics, fp=f"{m.model_dir}/{m.model_tag}/training.log")

    return log_metrics


def trainSMTScheduledSampling(m, dls, cfg: SMTConfig, save_model=True, final_eval=False):
    print("--- starting training setup ---")
    print(dls.info(), m.info(), sep="\n")
    optimiser = torch.optim.AdamW(m.parameters(), lr=cfg.learning_rate, weight_decay=cfg.w_decay)

    n_train = len(dls.idxs_train)
    n_valid = len(dls.idxs_valid)
    print(f"{n_train=}, {n_valid=}")

    epoch_iters = max(1, n_train // dls.batch_size)
    train_iters = cfg.train_epochs * epoch_iters
    eval_iters = max(1, n_valid // dls.batch_size)
    print(f"{train_iters=}, {eval_iters=} | n_epochs={cfg.train_epochs}, {epoch_iters=}")

    # lr scheduling: linear warmup then cosine decay
    warmup_iters = max(1, int(cfg.pct_warmup * train_iters))
    warmup = LinearLR(optimiser, start_factor=0.01, total_iters=warmup_iters)
    decay = CosineAnnealingLR(optimiser, T_max=train_iters - warmup_iters, eta_min=1e-6)
    scheduler = SequentialLR(optimiser, [warmup, decay], milestones=[warmup_iters])

    ss_prob = cfg.ss_max_probability
    ss_iters = train_iters - warmup_iters
    print(f"scheduled sampling: p=0 for {warmup_iters} iters; anneal to {ss_prob} over {ss_iters} iters")

    print("\n--- training model with scheduled sampling ---")
    m.train()
    idx_epoch = 0
    idx_epoch_batch = 0

    log_metrics = []
    with tqdm(range(train_iters), desc="training") as pbar:
        for iter in pbar:
            if idx_epoch_batch >= epoch_iters:
                idx_epoch += 1
                idx_epoch_batch = 0

            batch = dls.get_batch("train")
            stops_info, trip_context, dec_inputs, padding_mask, enc_targets, dec_targets = batch
            boarding_gt, alighting_gt = dec_inputs

            # compute sampling probability: 0 during warmup, linear ramp after
            if iter < warmup_iters:
                p = 0.0
            else:
                p = ss_prob * min(1.0, (iter - warmup_iters) / max(1, ss_iters))

            # pass 1 (no grad): get model predictions with ground-truth (GT) inputs (teacher forcing)
            if p > 0.0:
                with torch.no_grad():
                    forced = m(
                        stops_info, trip_context, dec_inputs, padding_mask, enc_targets, dec_targets
                    )

                # predictions are at decoder positions (shifted by 1 vs inputs)
                # pred[:, t] predicts target[:, t], which is input[:, t+1]
                # so pred[:, :-1] replaces input[:, 1:] (keep position 0 as GT always)
                bp = forced["boarding"][:, :-1].detach()  # (B, T-2)
                ap = forced["alighting"][:, :-1].detach()  # (B, T-2)
                ss_mask = torch.bernoulli(torch.full_like(bp, p)).bool()  # (B, T-2)

                # mix: replace GT with model preds where mask is True
                boarding_mixed = boarding_gt.clone()
                alighting_mixed = alighting_gt.clone()
                boarding_mixed[:, 1:][ss_mask] = bp[ss_mask]
                alighting_mixed[:, 1:][ss_mask] = ap[ss_mask]

                dec_inputs = (boarding_mixed, alighting_mixed)

            # pass 2 (with grad): forward with (possibly mixed) inputs
            result = m(stops_info, trip_context, dec_inputs, padding_mask, enc_targets, dec_targets)
            loss = result["loss"] if result["loss"] is not None else torch.tensor(0.0)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=cfg.max_grad_norm)
            optimiser.step()
            scheduler.step()
            idx_epoch_batch += 1

            if iter % cfg.eval_interval == 0 or iter == train_iters - 1:
                losses = estimateSMTLoss(m, dls, n_iters=eval_iters)
                log_metrics.append(losses)
                if (cfg.early_stopping == "valid") and lossEarlyStopping(log_metrics, m, cfg):
                    break
            elif log_metrics:
                losses = {"train": loss.item(), "valid": log_metrics[-1]["valid"]}
                log_metrics.append(losses)
                if (cfg.early_stopping == "train") and lossEarlyStopping(log_metrics, m, cfg):
                    break

            pf = {
                "epoch": f"{idx_epoch + 1}/{cfg.train_epochs}",
                "train_loss": f"{losses['train']:.4f}",
                "valid_loss": f"{losses['valid']:.4f}",
                "pSS": f"{p:.3f}",
            }
            pbar.set_postfix(pf)

    if cfg.early_stopping:
        m.load(m.model_tag, verbose=False)

    if final_eval:
        losses = estimateSMTLoss(m, dls, n_iters=eval_iters)
        log_metrics.append(losses)
        print(f"final evaluation: train loss {losses['train']:.4f} | valid loss {losses['valid']:.4f}")

    if save_model:
        m.save(verbose=True) if not cfg.early_stopping else None
        saveConfig(cfg, fp=f"{m.model_dir}/{m.model_tag}/config.json")
        saveTrainingLog(log_metrics, fp=f"{m.model_dir}/{m.model_tag}/training.log")

    return log_metrics


def trainGAEModelEpochs(
    matrices: torch.Tensor,
    cfg: SMTConfig,
    n_epochs: int,
    learning_rate: float | None = None,
) -> tuple[GraphAutoEncoder, torch.Tensor]:
    B, n_matrices, n_nodes, _ = matrices.shape
    learning_rate = learning_rate or cfg.learning_rate

    matrices = matrices.to(cfg.device)
    model = GraphAutoEncoder(n_nodes, cfg.embed_dim, n_matrices=n_matrices, device=cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    model.train()
    for epoch in tqdm(range(n_epochs), desc="training graph autoencoder"):
        optimizer.zero_grad()
        embeddings, reconstructed = model(matrices)
        loss = model.estimateGAELoss(matrices, reconstructed)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        embeddings, _ = model(matrices)
        embeddings = embeddings.squeeze(0)

    print(f"training loss history: {losses[0]:.4f} -> {losses[-1]:.4f}")
    return model, embeddings


@torch.no_grad()
def smtFinalEvaluation(
    m,
    dls,
    bundle: TransformBundle,
    subset: str,
    batch_size: int | None = 4096,
    raw_evaluation: bool = True,
    teacher_forcing: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert subset in ("train", "valid", "test"), "!!!"
    idxs_subset = getattr(dls, f"idxs_{subset}")
    assert len(idxs_subset) > 0, "!!!"

    batch_size = batch_size or dls.batch_size
    f_target = list(bundle.targetTransforms)
    l_dfs_preds = []
    l_dfs_targs = []
    was_training = m.training

    m.eval()
    for start in range(0, len(idxs_subset), batch_size):
        b_idxs = idxs_subset[start : start + batch_size]

        sids = dls.sl_sids[b_idxs].to(dls.device)
        ssarrival = dls.sl_ssarrival[b_idxs].to(dls.device)
        boarding_full = dls.sl_boarding[b_idxs].to(dls.device)
        alighting_full = dls.sl_alighting[b_idxs].to(dls.device)

        delay_targets = dls.sl_delay[b_idxs].to(dls.device)
        dwell_targets = dls.sl_dwell[b_idxs].to(dls.device)
        boarding_targets = boarding_full[:, 1:]
        alighting_targets = alighting_full[:, 1:]

        padding_mask = sids == dls.stoi[dls.PAD]
        gtc_cats = dls.gtc_cats[b_idxs].to(dls.device)
        gtc_conts = dls.gtc_conts[b_idxs].to(dls.device)

        if teacher_forcing:  # non-autoregressive decoding with full target context
            outputs = m(
                (sids, ssarrival),
                (gtc_cats, gtc_conts),
                (boarding_full[:, :-1], alighting_full[:, :-1]),
                padding_mask,
                (delay_targets, dwell_targets),
                (boarding_targets, alighting_targets),
            )
            boarding_preds = outputs["boarding"]
            alighting_preds = outputs["alighting"]
        else:  # autoregressive generation with greedy decoding
            # pass target-transformed fill values for autoregressive generation init
            B = sids.size(0)
            lffv = lambda c: dls.fill_values[c]
            lftf = lambda c: torch.full((B, 1), lffv(c), dtype=torch.float, device=dls.device)
            dec_init = (lftf("PC_Boarding"), lftf("PC_Alighting"))

            outputs = m.generate((sids, ssarrival), (gtc_cats, gtc_conts), dec_init, padding_mask)
            boarding_preds = outputs["boarding"][:, 1 : sids.size(1)]
            alighting_preds = outputs["alighting"][:, 1 : sids.size(1)]

        assert boarding_preds.shape == boarding_targets.shape, "!!!"
        assert alighting_preds.shape == alighting_targets.shape, "!!!"

        non_pad_mask = ~padding_mask
        rows = torch.arange(non_pad_mask.size(0), device=non_pad_mask.device)

        surr_mask = non_pad_mask.clone()
        surr_mask[:, 0] = False  # exclude first non-pad position (SOT)
        surr_mask[rows, non_pad_mask.sum(dim=1) - 1] = False  # exclude last non-pad position (EOT)

        pc_mask = non_pad_mask[:, 1:].clone()  # shift mask to align with PC predictions (exclude SOT)
        pc_mask[rows, pc_mask.sum(dim=1) - 1] = False  # exclude last non-pad position (EOT)

        lfnp = lambda t: t.detach().cpu().numpy()
        l_dfs_preds.append(
            pd.DataFrame(
                {
                    "PC_Boarding": lfnp(boarding_preds[pc_mask]),
                    "PC_Alighting": lfnp(alighting_preds[pc_mask]),
                    "ST_Delay": lfnp(outputs["delay"][surr_mask]),
                    "ST_Dwell": lfnp(outputs["dwell"][surr_mask]),
                }
            )
        )
        l_dfs_targs.append(
            pd.DataFrame(
                {
                    "PC_Boarding": lfnp(boarding_targets[pc_mask]),
                    "PC_Alighting": lfnp(alighting_targets[pc_mask]),
                    "ST_Delay": lfnp(delay_targets[surr_mask]),
                    "ST_Dwell": lfnp(dwell_targets[surr_mask]),
                }
            )
        )

    if was_training:
        m.train()

    preds = pd.concat(l_dfs_preds, ignore_index=True)
    targs = pd.concat(l_dfs_targs, ignore_index=True)
    metrics, comparison = evaluatePredictions(
        bundle, preds, targs, f_target=f_target, raw_evaluation=raw_evaluation
    )
    return metrics, comparison
