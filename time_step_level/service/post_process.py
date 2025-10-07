import numpy as np
from scipy.ndimage import binary_opening, binary_closing

def morphological_filter_1d(binary_signal, operation="closing", kernel_size=5):
    """
    Applies a 1D morphological operation (closing or opening) to a binary time series.
    
    Parameters
    ----------
    binary_signal : 1D numpy array of shape (n_samples,)
        The binary signal (values 0 or 1).
    operation : str, optional
        Which operation to apply: "closing" or "opening".
    kernel_size : int, optional
        Length of the 1D structuring element used for dilation/erosion.
        Larger values have a stronger smoothing/merging effect.
    
    Returns
    -------
    filtered_signal : 1D numpy array of shape (n_samples,)
        The binary signal after the morphological operation.
    """
    # Construct a 1D structuring element (all ones of length 'kernel_size')
    # For time-series, we treat this as a 1D "window."
    structure = np.ones(kernel_size, dtype=bool)
    
    # Apply the selected morphological operation
    if operation == "closing":
        # Closes small holes (0’s) inside regions of 1’s
        filtered = binary_closing(binary_signal, structure=structure)
    elif operation == "opening":
        # Removes small spurious 1’s (noise) that do not fit the structuring element
        filtered = binary_opening(binary_signal, structure=structure)
    else:
        raise ValueError("operation must be 'closing' or 'opening'")
    
    return filtered.astype(int)


def remove_short_events(binary_output, min_length, fs):
    """
    binary_output: 1D numpy array of shape (n_samples,)
    min_length: minimum length in seconds
    fs: sampling frequency (samples per second)
    """
    min_samples = int(min_length * fs)
    out = binary_output.copy()
    
    # Identify “on” regions
    is_seizure = False
    start_idx = 0
    
    for i in range(len(binary_output)):
        if not is_seizure and out[i] == 1:
            # transition from 0 to 1
            is_seizure = True
            start_idx = i
        elif is_seizure and (out[i] == 0 or i == len(binary_output)-1):
            # transition from 1 to 0, or end of array
            end_idx = i if out[i] == 0 else i+1
            length = end_idx - start_idx
            
            # If the region is too short, set it to 0
            if length < min_samples:
                out[start_idx:end_idx] = 0
            is_seizure = False
    
    return out