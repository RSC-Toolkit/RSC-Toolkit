from typing import ClassVar, Dict, Optional, List, Tuple, Type
from pydantic import BaseModel, field_validator, ConfigDict
import pathlib
import logging
import functools
import torch
import pandas as pd


from rsc_toolkit.datasets.base import BaseRSCDataset, BaseRSCDatasetDownloader, BaseRSCDatasetProcessor
from rsc_toolkit.datasets import DatasetFactory
from rsc_toolkit.datasets.download_utils import download_file, extract_zip


logger = logging.getLogger(__name__)


class ICBHIDatasetDownloader(BaseRSCDatasetDownloader):
    name: str = "ICBHI"
    dataset_download_url: str = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip"

    # Extra links on the page, maybe not needed.
    metadata_download_url: str = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_Challenge_demographic_information.txt"
    events_download_url: str = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/events.zip"

    @functools.cached_property
    def outputs_path(self) -> pathlib.Path:
        path = pathlib.Path(self.outputs_directory).expanduser().resolve()
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
    def from_path(cls, path: pathlib.Path):
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


class Cycle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio: torch.Tensor                # sliced waveform
    start: float                       # seconds
    end: float                         # seconds
    annotation: pd.Series              # the row from the annotation file
    label: Optional[int] = None        # processor fills this in
    features: Optional[torch.Tensor] = None  # e.g. spectrogram


class Recording(BaseModel):
    meta: RecordingMetadata
    wav_path: pathlib.Path
    txt_path: pathlib.Path

    @classmethod
    def from_paths(cls, wav: pathlib.Path, txt: pathlib.Path):
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

    def compute_cycles(self, sample_rate: int) -> List[Cycle]:
        """Slice waveform into Cycle objects according to annotations."""
        data = self._load_and_preprocess_audio(sample_rate)

        cycles = []
        for _, row in self.cycle_annotations.iterrows():
            start, end = row["Start"], row["End"]
            chunk = self._slice_data(start, end, data, sample_rate)
            cycles.append(
                Cycle(audio=chunk, start=start, end=end, annotation=row)
            )
        return cycles

    def _load_and_preprocess_audio(self, sample_rate: int):
        try:
            import torchaudio
            import torchaudio.transforms as T
            import librosa
        except ImportError:
            raise ImportError(
                "torchaudio and librosa are required for audio processing. "
                "Install with: pip install torchaudio librosa"
            )
        
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



class ICBHIDatasetProcessor(BaseRSCDatasetProcessor):
    name: str = "ICBHI"
    sample_rate: int = 10000 # TODO pass from config
    
    # Raw unpacked data here. Expecting other files in current dir.
    raw_data_directory: Optional[str] = None  # Configurable, defaults to None

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
        if self.raw_data_directory:
            return pathlib.Path(self.raw_data_directory)
        # Default fallback (for backward compatibility)
        default_path = pathlib.Path.home() / "research" / "RSC-Toolkit" / "downloaded_datasets" / "ICBHI_final_database" / "ICBHI_final_database"
        if default_path.exists():
            return default_path
        # If default doesn't exist, try relative to project
        project_root = pathlib.Path(__file__).parent.parent.parent.parent
        sample_data_path = project_root / "sample_data" / "toy_dataset" / "files"
        if sample_data_path.exists():
            logger.warning(f"Using toy dataset at {sample_data_path}")
            return sample_data_path
        raise ValueError(f"raw_data_directory not set and no default found. Please set raw_data_directory.")

    @property
    def metadata_folder(self):
        return (pathlib.Path(__file__).parent / "data").resolve()

    @property
    def metadata_file(self):
        return self.metadata_folder / "metadata.txt"

    def _discover_recordings(self):
        files = list(pathlib.Path(self.data_folder).iterdir())
        stems = {p.stem for p in files if p.suffix in {".wav", ".txt"}}

        pairs = []
        for stem in stems:
            wav = pathlib.Path(self.data_folder, stem + ".wav")
            txt = pathlib.Path(self.data_folder, stem + ".txt")
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
        # Normalize fields - use replace without inplace to avoid deprecation warning
        df = df.replace({
            "sex": {"M": 0, "F": 1},
            "chest_location": {"Tc": 0, "Al": 1, "Ar": 2, "Pl": 3, "Pr": 4, "Ll": 5, "Lr": 6}
        })
        return df

    def _attach_metadata(self, recordings: List[Recording], df: pd.DataFrame):
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

    def _extract_audio(self, sample) -> torch.Tensor:
        """Extract audio tensor from sample (Recording or Cycle)"""
        if isinstance(sample, Cycle):
            return sample.audio
        elif isinstance(sample, Recording):
            # Load full recording audio
            try:
                import torchaudio
            except ImportError:
                raise ImportError(
                    "torchaudio is required for audio processing. "
                    "Install with: pip install torchaudio"
                )
            data, _ = torchaudio.load(str(sample.wav_path))
            return data
        else:
            raise NotImplementedError(f"Cannot extract audio from {type(sample)}")
    
    def _get_sample_id(self, sample) -> str:
        """Get unique ID for sample (for caching)"""
        if isinstance(sample, Cycle):
            # Use recording path + cycle index
            rec = getattr(sample, '_recording', None)
            if rec:
                return f"{rec.wav_path.stem}_cycle_{sample.start:.3f}_{sample.end:.3f}"
            return f"cycle_{sample.start:.3f}_{sample.end:.3f}"
        elif isinstance(sample, Recording):
            return str(sample.wav_path.stem)
        else:
            return str(hash(str(sample)))

    def _extract_cycles(self, rec: Recording):
        print(rec.compute_cycles(sample_rate=self.sample_rate))
        # raw_cycles = rec.get_individual_cycles_torchaudio(
        #     annotations,
        #     self.config.audio_processing.sample_rate
        # )

        # rec.cycles = []
        # for audio_chunk, row in raw_cycles:
        #     label = self.map_label(row)   # processor-specific
        #     audio_chunk = cut_pad_sample_torchaudio(audio_chunk, self.config)
        #     rec.cycles.append((audio_chunk, label))

    def process_dataset(self) -> None:
        """
        Process the dataset for running
        """
        recordings = self._discover_recordings()
        logger.info(f"Found {len(recordings)} recordings")
        
        if len(recordings) == 0:
            logger.warning("No recordings found!")
            return
        
        # Load metadata if available
        try:
            df_metadata = self._load_metadata_file()
            self._attach_metadata(recordings=recordings, df=df_metadata)
            logger.debug(f"Attached metadata to {len(recordings)} recordings")
        except FileNotFoundError:
            logger.warning("Metadata file not found, skipping metadata attachment")
        except Exception as e:
            logger.warning(f"Error loading metadata: {e}, continuing without metadata")
        
        # Extract cycles from all recordings
        all_cycles = []
        for recording in recordings:
            try:
                cycles = recording.compute_cycles(sample_rate=self.sample_rate)
                all_cycles.extend(cycles)
            except Exception as e:
                logger.error(f"Error extracting cycles from {recording.wav_path}: {e}")
                continue
        
        logger.info(f"Extracted {len(all_cycles)} cycles from {len(recordings)} recordings")
        
        # Preprocess features if enabled
        if self.preprocess_features and self.feature_processor:
            logger.info(f"Preprocessing features with {self.feature_processor.__class__.__name__}")
            self.preprocess_all_features(all_cycles)
            logger.info("✅ Feature preprocessing complete!")
        else:
            logger.info("Feature preprocessing not enabled (preprocess_features=False or no feature_processor)")



@DatasetFactory.register_dataset_class("icbhi")
class ICBHIDataset(BaseRSCDataset):
    """
    ICBHI Dataset
    https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
    """
    downloader_class: ClassVar[Type[ICBHIDatasetDownloader]] = ICBHIDatasetDownloader
    processor_class: ClassVar[Type[ICBHIDatasetProcessor]] = ICBHIDatasetProcessor

