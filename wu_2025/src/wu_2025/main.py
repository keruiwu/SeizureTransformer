import torch
import numpy as np
from epilepsy2bids.annotations import Annotations
from epilepsy2bids.eeg import Eeg
from wu_2025.utils import load_models, get_dataloader, predict

def main(edf_file, outFile):
    eeg = Eeg.loadEdfAutoDetectMontage(edfFile=edf_file)
    assert eeg.montage is Eeg.Montage.UNIPOLAR, "Error: Only unipolar montages are supported."

    device = "cuda" if torch.cuda.is_available() else "cpu"

    fs = eeg.fs
    seq_len = eeg.data.shape[1]

    model = load_models(device)
    dataloader = get_dataloader(eeg.data, fs, batch_size=512)
    y_predict = predict(model, dataloader, device, seq_len)

    hyp = Annotations.loadMask(y_predict, eeg.fs)
    hyp.saveTsv(outFile)