"""Optional, additive model extensions used by the tuning experiments."""

from .pooling import GraphPoolSum


def apply_model_extensions(base, cfg):
    """Apply optional config-driven changes to a freshly built ChemPathGINE.

    Absent keys leave the model untouched (original behaviour). Supported:
      model.pool: "sum"   -> GraphPoolSum readout
      model.encoder: ...  -> alternative graph encoder (see encoders.py, if present)
    """
    model_cfg = getattr(cfg, "model", None)
    if str(getattr(model_cfg, "pool", "default")) == "sum":
        base.pool = GraphPoolSum(int(model_cfg.hidden_dim))
    encoder = str(getattr(model_cfg, "encoder", "gine"))
    if encoder != "gine":
        from .encoders import build_encoder  # noqa: WPS433 (optional module)

        base.encoder = build_encoder(encoder, cfg)
    return base


__all__ = ["GraphPoolSum", "apply_model_extensions"]
