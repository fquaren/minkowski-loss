"""
Tests for src/node_limits.py: our GPU only (by UUID), cores 4-11, on whichever node holds our
card; nothing elsewhere.
"""

import os

import pytest

from src import node_limits as nl


@pytest.fixture
def on_node(monkeypatch):
    # node32/node34 layout: the other group's card at PCI index 0, ours at 1
    monkeypatch.setattr(nl, "node_gpu_uuids", lambda: [nl.FORBIDDEN_GPU_UUID, nl.ALLOWED_GPU_UUID])
    monkeypatch.delenv("MINK_NODE_LIMITS", raising=False)


@pytest.fixture
def keep_affinity():
    saved = os.sched_getaffinity(0)
    yield
    os.sched_setaffinity(0, saved)


class TestGPU:
    def test_unset_defaults_to_gpu1_in_pci_order(self, on_node, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
        nl.enforce_gpu()
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
        assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"

    def test_index1_gets_pci_order(self, on_node, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
        nl.enforce_gpu()
        assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"

    @pytest.mark.parametrize("bad", ["0", "0,1", "1,0", nl.FORBIDDEN_GPU_UUID])
    def test_gpu0_refused(self, on_node, monkeypatch, bad):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", bad)
        with pytest.raises(nl.NodeLimitError):
            nl.enforce_gpu()

    def test_uuid_and_cpu_only_allowed(self, on_node, monkeypatch):
        for ok in (nl.ALLOWED_GPU_UUID, "", "-1"):
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ok)
            nl.enforce_gpu()

    def test_card_elsewhere_in_pci_order(self, monkeypatch):
        # Another node where our card sits at PCI index 0: the index follows the card.
        other = "GPU-00000000-0000-0000-0000-000000000000"
        monkeypatch.setattr(nl, "node_gpu_uuids", lambda: [nl.ALLOWED_GPU_UUID, other])
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        nl.enforce_gpu()
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")   # a launcher's hard-coded 1 is refused there
        with pytest.raises(nl.NodeLimitError):
            nl.enforce_gpu()

    def test_off_node_is_noop(self, monkeypatch):
        # A host without our card (node34 since 2026-11-03, a laptop): nothing is touched.
        monkeypatch.setattr(nl, "node_gpu_uuids", lambda: ["GPU-1765684b-5b60-4027-4bda-04b926a7bcdb"])
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        nl.enforce()                      # must not raise
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


class TestCPU:
    def test_pins_to_allowed(self, on_node, keep_affinity):
        os.sched_setaffinity(0, set(range(os.cpu_count())))
        got = nl.enforce_cpu()
        assert set(os.sched_getaffinity(0)) == set(got) <= set(nl.ALLOWED_CPUS)
        assert len(got) <= 8

    def test_keeps_a_narrower_allowed_set(self, on_node, keep_affinity):
        os.sched_setaffinity(0, {4, 5})
        assert nl.enforce_cpu() == {4, 5}

    def test_disjoint_set_moves_into_allowed(self, on_node, keep_affinity):
        os.sched_setaffinity(0, {0, 1})
        assert nl.enforce_cpu() == nl.ALLOWED_CPUS

    def test_leaves_four_cores_free(self):
        assert len(set(range(12)) - nl.ALLOWED_CPUS) >= 4
