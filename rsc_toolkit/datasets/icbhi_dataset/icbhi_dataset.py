from typing import ClassVar, Dict, Optional, List, Tuple
from pathlib import Path
from enum import Enum
import logging
import functools

from pydantic import BaseModel, field_validator, ConfigDict, Field
import torch
import pandas as pd
import tqdm

from rsc_toolkit.datasets.base import BaseRSCDataset, BaseRSCDatasetDownloader, BaseRSCDatasetProcessor
from rsc_toolkit.datasets import DatasetFactory
from rsc_toolkit.datasets.cycle_store import BaseCycleStore, FileCycleStore
from rsc_toolkit.datasets.download_utils import download_file, extract_zip


logger = logging.getLogger(__name__)


class ICBHIDatasetDownloader(BaseRSCDatasetDownloader):
    name: str = "ICBHI"
    dataset_download_url: str = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip"

    # Extra links on the page, maybe not needed.
    metadata_download_url: str = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_Challenge_demographic_information.txt"
    events_download_url: str = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/events.zip"

    @functools.cached_property
    def outputs_path(self) -> Path:
        path = Path(self.outputs_directory).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def zip_path(self):
        return self.outputs_path / "ICBHI_final_database.zip"

    @property
    def extracted_dir(self):
        return self.outputs_path / "ICBHI_final_database"

    def validate_dataset_files(self) -> bool:
        """
        Validate the dataset files.
        """
        if not self.extracted_dir.exists():
            # this is a really simple check. Maybe better to see that all expected files are downloaded and unpacked.
            return False
        return True

    def download_dataset(self) -> None:
        """Download and extract the ICBHI dataset."""
        logger.info(
            "Preparing to download ICBHI Dataset.\n" \
            "Process:\n" \
            f"\t- Download and unzip single archive from {self.dataset_download_url}\n"
            )

        if self.force or not self.zip_path.exists():
            download_file(url=self.dataset_download_url, out_path=self.zip_path)
        else:
            logger.info(f"Found dataset at {self.zip_path}, skipping download!")

        if self.force or not self.extracted_dir.exists():
            extract_zip(zip_path=self.zip_path, out_dir=self.extracted_dir)
        else:
            logger.info(f"Found extracted files at {self.extracted_dir}, skipping extraction!")


DEVICE_IDS = {
    "Meditron": 0,
    "LittC2SE": 1,
    "Litt3200": 2,
    "AKGC417L": 3,
}

CHEST_LOCATIONS = {
    "Tc": "Trachea",
    "Al": "Anterior left",
    "Ar": "Anterior right",
    "Pl": "Posterior left",
    "Pr": "Posterior right",
    "Ll": "Lateral left",
    "Lr": "Lateral right",
}

class RespiratoryLabel(Enum):
    NORMAL = 0
    CRACKLE = 1
    WHEEZE = 2
    BOTH = 3


LABEL_MAP = { # (Crackle, Wheeze)
    (0, 0): 0,  # normal
    (1, 0): 1,  # crackle only
    (0, 1): 2,  # wheeze only
    (1, 1): 3,  # both crackle and wheeze
}


class RecordingMetadata(BaseModel):
    # From filename
    patient_id: int
    recording_index: str
    chest_location: str
    acquisition_mode: str
    device: str
    ext: str

    # From metadata.txt, added afterwards
    age: Optional[float] = None
    sex: Optional[int] = None  # 0=Male, 1=Female
    adult_BMI: Optional[float] = None
    child_weight: Optional[float] = None
    child_height: Optional[float] = None
    chest_location_id: Optional[int] = None

    @field_validator("patient_id", mode="before")
    def validate_patient_id(cls, v):
        return int(v) if isinstance(v, str) else v

    @field_validator("chest_location", mode="before")
    def normalize_chest_location(cls, v):
        return CHEST_LOCATIONS.get(v, v)

    @field_validator("device", mode="before")
    def validate_device(cls, v):
        if v not in DEVICE_IDS:
            raise ValueError(f"Unknown device: {v}")
        return v

    @property
    def device_id(self) -> int:
        return DEVICE_IDS[self.device]

    @classmethod
    def from_path(cls, path: Path):
        parts = path.stem.split("_")
        if len(parts) != 5:
            raise ValueError(f"Unexpected filename format: {path.name}")
        patient_id, rec_idx, chest, acq, device = parts
        return cls(
            patient_id=patient_id,
            recording_index=rec_idx,
            chest_location=chest,
            acquisition_mode=acq,
            device=device,
            ext=path.suffix,
        )


class ICBHICycle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio: torch.Tensor                # sliced waveform
    start: float                       # seconds
    end: float                         # seconds
    label: RespiratoryLabel
    meta: RecordingMetadata
    features: Optional[torch.Tensor] = None  # e.g. spectrogram


class Recording(BaseModel):
    meta: RecordingMetadata
    wav_path: Path
    txt_path: Path

    @classmethod
    def from_paths(cls, wav: Path, txt: Path):
        base = RecordingMetadata.from_path(wav)
        return cls(meta=base, wav_path=wav, txt_path=txt)

    @functools.cached_property
    def cycle_annotations(self) -> pd.DataFrame:
        """
        Load annotations from the .txt file and store them.
        
        Looks like:
        0.036	0.579	0	0
        0.579	2.45	0	0
        2.45	3.893	0	0
        """
        df = pd.read_csv(
            self.txt_path,
            delimiter="\t",
            header=None
        )
        df.columns = ["Start", "End", "Crackle", "Wheeze"] # TODO verify order
        return df

    def compute_cycles(self, sample_rate: int) -> list[ICBHICycle]:
        """Slice waveform into Cycle objects according to annotations."""
        data = self._load_and_preprocess_audio(sample_rate)

        cycles = []
        for _, row in self.cycle_annotations.iterrows():
            start, end = row["Start"], row["End"]
            c = int(row["Crackle"])
            w = int(row["Wheeze"])
            label = {
                (0, 0): RespiratoryLabel.NORMAL,
                (1, 0): RespiratoryLabel.CRACKLE,
                (0, 1): RespiratoryLabel.WHEEZE,
                (1, 1): RespiratoryLabel.BOTH,
            }[(c, w)]
            chunk = self._slice_data(start, end, data, sample_rate)
            cycles.append(
                ICBHICycle(
                    audio=chunk,
                    start=start,
                    end=end,
                    label=label,
                    meta=self.meta,
                )
            )
        return cycles

    def _load_and_preprocess_audio(self, sample_rate: int):
        import torchaudio, torchaudio.transforms as T, librosa
        sr = librosa.get_samplerate(str(self.wav_path))
        data, _ = torchaudio.load(str(self.wav_path))
        if sr != sample_rate:
            data = T.Resample(sr, sample_rate)(data)
        fade_samples = int(sample_rate / 16)
        data = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape="linear")(data)
        return data

    def _slice_data(self, start: float, end: float, data, sample_rate: int):
        max_ind = data.shape[1]
        start_ind = min(int(start * sample_rate), max_ind)
        end_ind = min(int(end * sample_rate), max_ind)
        return data[:, start_ind:end_ind]



class ICBHIProcessor(BaseRSCDatasetProcessor):
    
    # Raw unpacked data here. Expecting other files in current dir.
    raw_data_directory: Path

    name: str = "ICBHI"
    sample_rate: int = 10000 # TODO pass from config
    
    DEVICE_IDS: Dict[str, int] = {'Meditron': 0, 'LittC2SE': 1, 'Litt3200': 2, 'AKGC417L': 3}

    @property
    def train_cache_file(self):
        return self.cache_dir_path / "icbhi_training.pt"

    @property
    def train_cache_classes_file(self):
        return self.cache_dir_path / "icbhi_class_nums.pt"

    @property
    def test_cache_file(self):
        return self.cache_dir_path / "icbhi_test.pt"
    
    def validate_dataset_files(self) -> bool:
        """
        Validate preocessed files files
        """
        if not self.train_cache_classes_file.exists():
            return False
        if not self.train_cache_file.exists():
            return False
        if not self.test_cache_file.exists():
            return False
        return True

    @property
    def data_folder(self):
        return Path(self.raw_data_directory)

    @property
    def metadata_folder(self):
        return (Path(__file__).parent / "data").resolve()

    @property
    def metadata_file(self):
        return self.metadata_folder / "metadata.txt"

    def _discover_recordings(self):
        files = list(Path(self.data_folder).iterdir())
        stems = {p.stem for p in files if p.suffix in {".wav", ".txt"}}

        pairs = []
        for stem in stems:
            wav = Path(self.data_folder, stem + ".wav")
            txt = Path(self.data_folder, stem + ".txt")
            if wav.exists() and txt.exists():
                pairs.append(Recording.from_paths(wav, txt))
            else:
                # skip incomplete pairs or log a warning
                logger.warning(f"Missing pair for {stem}")
        return pairs
    
    def _load_metadata_file(self) -> pd.DataFrame:
        meta_path = self.metadata_file
        df = pd.read_csv(
            meta_path,
            names=["age", "sex", "adult_BMI", "child_weight", "child_height", "chest_location"],
            delimiter="\t",
            index_col=0  # patient_id is the index
        )
        # Normalize fields (safe for pandas 3.0+)
        df.replace({
            "sex": {"M": 0, "F": 1},
            "chest_location": {"Tc": 0, "Al": 1, "Ar": 2, "Pl": 3, "Pr": 4, "Ll": 5, "Lr": 6}
        }, inplace=True)
        return df

    def _attach_metadata(self, recordings: list[Recording], df: pd.DataFrame):
        for rec in recordings:
            if rec.meta.patient_id not in df.index:
                logger.warning(f"No metadata found for patient {rec.meta.patient_id}")
                continue

            row = df.loc[rec.meta.patient_id]
            rec.meta.age = row["age"]
            rec.meta.sex = row["sex"]
            rec.meta.adult_BMI = row["adult_BMI"]
            rec.meta.child_weight = row["child_weight"]
            rec.meta.child_height = row["child_height"]
            rec.meta.chest_location_id = row["chest_location"]

    def _extract_cycles(self, rec: Recording, store: BaseCycleStore):
        cycles = rec.compute_cycles(sample_rate=self.sample_rate)
        # for i, cycle in enumerate(tqdm.tqdm(cycles, desc="Processing cycles")): # Too much info in terminal
        for i, cycle in enumerate(cycles):
            cycle_id = f"{rec.wav_path.stem}_cycle_{i:03d}"
            store.add_cycle(key=cycle_id, cycle=cycle)
        
    def process_dataset(self) -> None:
        """
        Process the dataset for running
        """
        recordings = self._discover_recordings()
        if len(recordings) == 0:
            raise FileNotFoundError("Could not find any samples.")
        logger.debug(f"Found {len(recordings)} recordings: Example {recordings[0]}")
        
        df_metadata = self._load_metadata_file()
        self._attach_metadata(recordings=recordings, df=df_metadata)
        logger.debug(f"Attached metadata: Example {recordings[0]}")
        
        for recording in tqdm.tqdm(recordings, desc="Processing recordings"):
            self._extract_cycles(recording, store=self.cycle_store)




@DatasetFactory.register_dataset_class("icbhi")
class ICBHIDataset(BaseRSCDataset):
    """
    ICBHI Dataset
    https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
    """
    downloader_class: ClassVar[type[ICBHIDatasetDownloader]] = ICBHIDatasetDownloader
    processor_class: ClassVar[type[ICBHIProcessor]] = ICBHIProcessor

