import pytest
import torch


from rsc_toolkit.datasets.cycle_store import Cycle, FileCycleStore


class TestCycle(Cycle):

    def to_index_entry(self) -> dict:
        return self.model_dump(exclude=["audio"])


@pytest.fixture
def dummy_cycle():
    """Helper to make a minimal valid Cycle."""
    audio = torch.randn(1, 1000)  # 1-channel 1000-sample waveform
    c = TestCycle(audio=audio, start=0.0, end=1.0)
    return c


def test_cycle_save_and_load(dummy_cycle, tmp_path):
    store = FileCycleStore(root=tmp_path)
    
    cycle_id = "dummy"

    store.save_cycle(
        cycle_id=cycle_id,
        cycle=dummy_cycle,
    )
    loaded_cycle = store.load_cycle(
        cycle_id=cycle_id,
        cls=TestCycle,
    )
    
    # Compare scalar fields
    assert dummy_cycle.start == loaded_cycle.start
    assert dummy_cycle.end == loaded_cycle.end

    # Compare tensor fields
    assert torch.allclose(dummy_cycle.audio, loaded_cycle.audio)


def test_cycle_index(dummy_cycle, tmp_path):
    store = FileCycleStore(root=tmp_path)
    
    store.save_cycle(
        cycle_id="1",
        cycle=dummy_cycle,
    )
    store.save_cycle(
        cycle_id="2",
        cycle=dummy_cycle,
    )
    assert len(store.finalize_index()) == 2
    