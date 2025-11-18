from pydantic import PrivateAttr
import pytest
import pathlib


from rsc_toolkit.datasets.icbhi_dataset.icbhi_dataset import ICBHIProcessor, ICBHICycle
from rsc_toolkit.datasets.cycle_store import FileCycleStore

from ...common.path_utils import fixture_path


def test_toy_dataset_processing(tmp_path):

    cycle_store = FileCycleStore(path=tmp_path / "icbhi" / "cycles")
    
    processor = ICBHIProcessor(
        raw_data_directory=fixture_path("toy_dataset") / "files",
        outputs_directory=tmp_path / "ichbi",
        cycle_store=cycle_store,
    )
    
    processor.process_dataset()
    

    print(cycle_store.get_cycle(list(cycle_store.index)[0], cls=ICBHICycle))
