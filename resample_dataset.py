import itertools
import librosa
import soundfile as sf
from pathlib import Path


TARGET_SAMPLE_RATE = 32_000


def resample(path, target_sr):
    y, sr = librosa.load(path)
    r = librosa.resample(y=y, orig_sr=sr, target_sr=target_sr)
    sf.write(path, data=r, samplerate=target_sr)
    
    
if __name__ == "__main__":
    datafiles = Path("dataset/phonk")
    train_soundfiles = [f for f in (datafiles/"train").iterdir() if f.suffix != ".json"]
    valid_soundfiles = [f for f in (datafiles/"valid").iterdir() if f.suffix != ".json"]
    
    for file in itertools.chain(train_soundfiles, valid_soundfiles):
        resample(str(file), TARGET_SAMPLE_RATE)