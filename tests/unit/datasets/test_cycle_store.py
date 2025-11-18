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
    path = tmp_path / "cyclestore"
    store = FileCycleStore(path=path)
    
    key = store.add_cycle(
        cycle=dummy_cycle,
    )

    loaded_cycle = store.get_cycle(
        key=key,
        cls=TestCycle,
    )
    
    # Compare scalar fields
    assert dummy_cycle.start == loaded_cycle.start
    assert dummy_cycle.end == loaded_cycle.end

    # Compare tensor fields
    assert torch.allclose(dummy_cycle.audio, loaded_cycle.audio)


def test_cycle_index(dummy_cycle, tmp_path):
    path = tmp_path / "cyclestore"
    store = FileCycleStore(path=path, allow_duplicates=True)
    store.add_cycle(
        cycle=dummy_cycle,
    )
    store.add_cycle(
        cycle=dummy_cycle,
    )
    assert len(store.index) == 2
    

def test_cycle_index_from_disk(dummy_cycle, tmp_path):
    path = tmp_path / "cyclestore"
    store = FileCycleStore(path=path, allow_duplicates=True)
    store.add_cycle(
        cycle=dummy_cycle,
    )
    store.add_cycle(
        cycle=dummy_cycle,
    )

    loaded_store = FileCycleStore(path=path, allow_duplicates=True)
    assert len(loaded_store.index) == 2
    

def test_duplicate_raises_error(dummy_cycle, tmp_path):
    path = tmp_path / "cyclestore"
    store = FileCycleStore(path=path, allow_duplicates=False)
    store.add_cycle(
        cycle=dummy_cycle,
    )
    with pytest.raises(ValueError):
        store.add_cycle(
            cycle=dummy_cycle,
        )
