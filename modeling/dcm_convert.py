import numpy as np
from datetime import datetime
import pydicom
from pydicom.dataset import FileMetaDataset, Dataset


def convert_npy_to_dcm(
    npy_array: np.ndarray,
    output_dcm_path: str,
) -> None:
    """Converts a numpy array to a DICOM file.
    Args:
        npy_array (np.ndarray): numpy array (C, H, W)
        output_dcm_path (str): output path for the DICOM file
    Returns:
        None
    """
    
    ds = Dataset()
    metadata = FileMetaDataset()
    metadata.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.file_meta = metadata
    
    # need for make pdf
    ds.PatientName = "Hong Gil Dong"
    ds.PatientID = "123456"  
    ds.PatientSex = "M"  
    ds.PatientAge = "30"  
    ds.PatientBirthDate = datetime.strptime("19940101", "%Y%m%d").date() 
    
    # need for convert
    ds.StudyDescription = "Knee MRI Dataset"
    ds.Rows, ds.Columns = npy_array.shape[1:] # H, W
    ds.NumberOfFrames = npy_array.shape[0]  # C
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]  # Assuming slices along the z-axis
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1 # 1 for grayscale, 3 for RGB
    ds.PhotometricInterpretation = "MONOCHROME2"
    
    ds.PlanarConfiguration = 1  # 0이면 한 이미지에 모두, 1이면 Channel 별로
    ds.SeriesDescription = "44 Channel MRI Series"
    ds.Modality = "MR"
    
    # dcm은 default가 uint16
    ds.PixelData = npy_array.astype(np.uint16).tobytes()
    ds.save_as(output_dcm_path)
    print(f"Saved the DICOM file at {output_dcm_path}")


def read_dicom_info(dicom_file_path):
    ds = pydicom.dcmread(dicom_file_path, force=True)
    
    print("DICOM Information:")
    print(f"Patient Name: {ds.PatientName}")
    print(f"Patient ID: {ds.PatientID}")
    print(f"Patient Age: {ds.PatientAge}")
    print(f"Patient Sex: {ds.PatientSex}")
    print(f"Patient Birth Date: {ds.PatientBirthDate}")
    
    
def convert_dcm_to_numpy(dcm_path: str) -> list[dict, np.ndarray]:
    """Converts a DICOM file to a numpy array.
    Args:
        dcm_path (str): path to the DICOM file
    Returns:
        list[dict, np.ndarray]: 
            - dict: metadata of the DICOM file
            - np.ndarray: numpy array (C, H, W)
    """
    ds = pydicom.dcmread(dcm_path, force=True)
    ds_metadata = {
        "PatientName": ds.PatientName,
        "PatientID": ds.PatientID,
        "PatientSex": ds.PatientSex,
        "PatientAge": ds.PatientAge,
        "PatientBirthDate": ds.PatientBirthDate,
    }    
    npy_array = ds.pixel_array.astype(np.uint8)
    
    return ds_metadata, npy_array


if __name__ == '__main__':
    raw_path = "C:/Users/zeroone/Downloads/raw_dataset/train/axial/0000.npy"
    raw_np_img = np.load(raw_path)
    
    output_dcm_path = "output_dicom.dcm"
    convert_npy_to_dcm(raw_np_img, output_dcm_path)
    read_dicom_info(output_dcm_path)
    metadata, npy_array = convert_dcm_to_numpy(output_dcm_path)
    # print(metadata)
    