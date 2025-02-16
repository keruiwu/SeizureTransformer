# SeizureTransformer

## Abstract
We introduce a novel deep-learning architecture for simultaneous seizure detection that departs from traditional window-level classification methods. Instead of assigning a single label to segmented windows, our approach utilizes a deep encoder, comprising 1D convolutions, Network-in-Network modules, residual connections, and a transformer encoder with global attention, to process raw EEG signals in the time domain. This design produces high-level representations that capture rich temporal dependencies at each time step. A streamlined decoder then converts these features into a sequence of probabilities, directly indicating the presence or absence of seizures at every time step. By operating at the time-step level, our method avoids the need for complex post-processing to map window labels to events and eliminates redundant overlapping inferences, thereby enhancing the model’s efficiency and accuracy. Extensive experiments on a public EEG seizure detection dataset demonstrate that our model significantly outperforms existing approaches, underscoring its potential for real-time, precise seizure detection.

## Poster
[poster link](https://drive.google.com/drive/folders/17pKhwFc4x1_2zwXTndKawoNKlaXIW-VE?usp=sharing)

## Instruction
1. Download the model's weight from [here](https://drive.google.com/drive/folders/17pKhwFc4x1_2zwXTndKawoNKlaXIW-VE?usp=sharing)
2. Put the weight file in **wu_2025/src/wu_2025**
