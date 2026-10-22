import pandas as pd
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
from rdkit import Chem

from chempath_gine.data import MoleculeDataset, collate_molecules
from chempath_gine.extensions import apply_model_extensions
from chempath_gine.extensions.encoders import (
    AttentiveFPEncoder,
    DMPNNEncoder,
    build_encoder,
    reverse_edge_index,
)
from chempath_gine.models import ChemPathGINE
from chempath_gine.models.gine_encoder import GINEEncoder
from chempath_gine.training.train import _ns
from chempath_gine.uncertainty import GaussianGINE
from chempath_gine.uncertainty.attribution import integrated_gradients, readout_attention

HIDDEN = 32
SMILES = ["CCO", "C", "c1ccccc1O", "O=C1c2ccccc2C(=O)c2ccccc21", "CC(=O)N"]
SINGLE_ATOM = 1  # index of "C" (methane: one heavy atom, self-loop sentinel edge)


@pytest.fixture(scope="module")
def toy_ds(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("enc")
    csv = tmp_path / "toy.csv"
    pd.DataFrame(
        {
            "SMILES": SMILES,
            "Solubility": [0.1, -0.9, -0.2, -5.2, 1.0],
            "MolWt": [46.0, 16.0, 94.0, 208.0, 59.0],
        }
    ).to_csv(csv, index=False)
    return MoleculeDataset(
        csv,
        mode="train",
        smiles_col="SMILES",
        target_col="Solubility",
        descriptor_cols=["MolWt"],
        scaler_dir=tmp_path / "scalers",
        scaler_key="toy",
        cache_dir=tmp_path / "cache",
        skip_path_features=True,
    )


def _cfg(encoder=None, heads=4, layers=2):
    model = {
        "variant": "gine_desc",
        "hidden_dim": HIDDEN,
        "num_gnn_layers": layers,
        "num_attention_layers": 1,
        "heads": heads,
        "dropout": 0.0,
    }
    if encoder is not None:
        model["encoder"] = encoder
    return _ns({"model": model})


def _base(ds, cfg):
    m = cfg.model
    return ChemPathGINE(
        hidden_dim=m.hidden_dim, num_gnn_layers=m.num_gnn_layers,
        num_attention_layers=m.num_attention_layers, heads=m.heads, dropout=m.dropout,
        path_dim=ds.path_dim, descriptor_dim=ds.descriptor_dim, max_path_distance=12,
        variant=m.variant,
    )


def _gaussian(ds, encoder, seed=0):
    torch.manual_seed(seed)
    cfg = _cfg(encoder)
    return GaussianGINE(apply_model_extensions(_base(ds, cfg), cfg)).eval()


def _batch(ds, idx=None):
    idx = range(len(ds)) if idx is None else idx
    return collate_molecules([ds[i] for i in idx])


def test_toy_set_contains_single_atom_sentinel(toy_ds):
    data = toy_ds[SINGLE_ATOM]
    assert data.num_nodes == 1
    assert data.edge_index.tolist() == [[0], [0]]


@pytest.mark.parametrize("name,cls", [("dmpnn", DMPNNEncoder), ("attentivefp", AttentiveFPEncoder)])
def test_encoder_node_state_shape(toy_ds, name, cls):
    torch.manual_seed(0)
    enc = build_encoder(name, _cfg(name)).eval()
    assert isinstance(enc, cls)
    batch = _batch(toy_ds)
    with torch.no_grad():
        h = enc(batch)
    assert h.shape == (batch.num_nodes, HIDDEN)
    assert torch.isfinite(h).all()
    single = _batch(toy_ds, [SINGLE_ATOM])
    with torch.no_grad():
        h1 = enc(single)
    assert h1.shape == (1, HIDDEN)
    assert torch.isfinite(h1).all()


@pytest.mark.parametrize("name", ["dmpnn", "attentivefp"])
def test_gaussian_forward_with_alternative_encoder(toy_ds, name):
    model = _gaussian(toy_ds, name)
    assert not isinstance(model.base.encoder, GINEEncoder)
    batch = _batch(toy_ds)
    with torch.no_grad():
        mu, logvar = model(batch)
        z, mu_hf = model.head_features(batch)
        # Attribution split must reproduce forward exactly.
        h0, e0 = model.embed(batch)
        h = model.node_states_from_embeddings(h0, e0, batch.edge_index)
        mu_split, _ = model.readout(h, batch.batch, batch.desc)
    n = len(SMILES)
    assert mu.shape == (n,) and logvar.shape == (n,)
    assert z.shape == (n, HIDDEN)
    assert torch.isfinite(mu).all()
    assert torch.allclose(logvar, torch.zeros_like(logvar))  # sigma = 1 at init
    assert torch.equal(mu, mu_hf)
    assert torch.allclose(mu, mu_split, atol=1e-6)
    # The single-atom molecule alone gives the same prediction as inside the batch.
    with torch.no_grad():
        mu1, _ = model(_batch(toy_ds, [SINGLE_ATOM]))
    assert mu1.shape == (1,)
    assert torch.allclose(mu1, mu[SINGLE_ATOM : SINGLE_ATOM + 1], atol=1e-5)


@pytest.mark.parametrize("name", ["dmpnn", "attentivefp"])
def test_alternative_encoder_trains(toy_ds, name):
    torch.manual_seed(0)
    cfg = _cfg(name)
    cfg.model.dropout = 0.1
    model = GaussianGINE(apply_model_extensions(_base(toy_ds, cfg), cfg), detach_variance_input=True)
    model.train()
    batch = _batch(toy_ds)
    mu, logvar = model(batch)
    loss = ((mu - batch.y.view(-1)) ** 2).mean() + logvar.mean()
    loss.backward()
    enc_grads = [p.grad for p in model.base.encoder.parameters() if p.requires_grad]
    assert enc_grads and all(g is not None and torch.isfinite(g).all() for g in enc_grads)
    assert any(float(g.abs().sum()) > 0 for g in enc_grads)
    # Nothing from the discarded GINE encoder is left in the model.
    assert not any(isinstance(m, GINEEncoder) for m in model.modules())


def test_reverse_edge_index_across_batch_with_sentinel(toy_ds):
    # The sentinel (odd edge count) sits in the middle, so a pairwise e ^ 1 rule would break.
    batch = _batch(toy_ds, [0, SINGLE_ATOM, 2])
    rev, found = reverse_edge_index(batch.edge_index, batch.num_nodes)
    assert bool(found.all())
    src, dst = batch.edge_index
    assert torch.equal(src[rev], dst) and torch.equal(dst[rev], src)
    loops = (src == dst).nonzero().view(-1)
    assert loops.numel() == 1 and int(rev[loops[0]]) == int(loops[0])
    # A directed edge without a reverse is flagged, not silently paired.
    rev2, found2 = reverse_edge_index(torch.tensor([[0, 1, 1], [1, 0, 2]]), 3)
    assert found2.tolist() == [True, True, False]
    assert rev2.tolist() == [1, 0, 2]


def test_dmpnn_single_atom_has_no_incoming_message():
    # One atom with the self-loop sentinel: the reverse of 0->0 is itself, so the
    # aggregated message is exactly zero and msg stays at msg0 + layer(msg0, 0).
    torch.manual_seed(0)
    enc = DMPNNEncoder(HIDDEN, 2, 0.0).eval()
    h0 = torch.randn(1, HIDDEN)
    e0 = torch.randn(1, HIDDEN)
    ei = torch.tensor([[0], [0]])
    with torch.no_grad():
        out = enc.node_states_from_embeddings(h0, e0, ei)
        msg0 = enc.edge_init(torch.cat([h0, e0], dim=-1))
        msg = msg0
        for layer in enc.msg_layers:
            msg = msg + layer(torch.cat([msg0, torch.zeros_like(msg0)], dim=-1))
        ref = h0 + torch.nn.functional.silu(enc.norm(enc.node_update(torch.cat([h0, msg], dim=-1))))
    assert torch.allclose(out, ref, atol=1e-6)


@pytest.mark.parametrize("name", ["dmpnn", "attentivefp"])
def test_encoder_invariant_to_edge_storage_order(toy_ds, name):
    torch.manual_seed(0)
    enc = build_encoder(name, _cfg(name)).eval()
    batch = _batch(toy_ds)
    with torch.no_grad():
        h0, e0 = enc.embed(batch)
        ref = enc.node_states_from_embeddings(h0, e0, batch.edge_index)
        perm = torch.randperm(batch.edge_index.size(1), generator=torch.Generator().manual_seed(3))
        out = enc.node_states_from_embeddings(h0, e0[perm], batch.edge_index[:, perm])
    assert torch.allclose(out, ref, atol=1e-5)


def test_gine_path_unchanged_vs_chempath_gine(toy_ds):
    batch = _batch(toy_ds)
    for encoder in (None, "gine"):
        torch.manual_seed(0)
        cfg = _cfg(encoder)
        base = _base(toy_ds, cfg)
        reference = _base(toy_ds, cfg)
        reference.load_state_dict({k: v.clone() for k, v in base.state_dict().items()})
        model = GaussianGINE(apply_model_extensions(base, cfg)).eval()
        reference.eval()
        assert type(model.base.encoder) is GINEEncoder
        with torch.no_grad():
            mu, logvar = model(batch)
            ref = reference(batch)
            h0, e0 = model.embed(batch)
            h = model.node_states_from_embeddings(h0, e0, batch.edge_index)
            mu_split, _ = model.readout(h, batch.batch, batch.desc)
            z, mu_hf = model.head_features(batch)
        assert torch.allclose(mu, ref, atol=1e-6)
        assert torch.equal(mu, mu_split)
        assert torch.equal(mu, mu_hf)
        assert torch.equal(h, reference.encoder(batch))
        assert torch.allclose(logvar, torch.zeros_like(logvar))


@pytest.mark.parametrize("name", ["dmpnn", "attentivefp"])
def test_attribution_works_for_alternative_encoders(toy_ds, name):
    model = _gaussian(toy_ds, name, seed=1)
    for p in model.parameters():
        p.requires_grad_(False)
    for i in (0, SINGLE_ATOM, 3):
        data = toy_ds[i]
        mol = Chem.MolFromSmiles(data.smiles)
        ig = integrated_gradients(model, data, n_bonds=mol.GetNumBonds(), steps=256)
        assert abs(ig["completeness_gap"]) < 0.02 * max(1.0, abs(ig["delta"]))
        attn = readout_attention(model, data)
        assert attn.shape == (mol.GetNumAtoms(),)
        assert abs(attn.sum() - 1.0) < 1e-5


def test_attribution_raises_for_encoder_without_split(toy_ds):
    class PlainEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = DMPNNEncoder(HIDDEN, 1, 0.0)

        def forward(self, batch):
            return self.inner(batch)

    torch.manual_seed(0)
    cfg = _cfg()
    base = _base(toy_ds, cfg)
    base.encoder = PlainEncoder()
    model = GaussianGINE(base).eval()
    batch = _batch(toy_ds)
    with torch.no_grad():
        mu, _ = model(batch)  # the generic forward only needs encoder(batch)
    assert mu.shape == (len(SMILES),)
    with pytest.raises(TypeError, match="embed"):
        model.embed(batch)
    with pytest.raises(TypeError, match="node_states_from_embeddings"):
        model.node_states_from_embeddings(torch.zeros(1, HIDDEN), torch.zeros(1, HIDDEN),
                                          torch.zeros(2, 1, dtype=torch.long))


def test_build_encoder_errors():
    with pytest.raises(ValueError, match="model.encoder"):
        build_encoder("schnet", _cfg())
    with pytest.raises(ValueError, match="divisible"):
        build_encoder("attentivefp", _cfg(heads=5))
