"""
encoding_mappings.py — Column encoding mappings for the oncology reference
dataset (see notbook/test_recur.ipynb), for use with
ml_framework.preprocessing.encoding.encode_dataframe().

Moved here verbatim from what used to be hardcoded defaults inside
encode_dataframe() itself — same mappings, same column names, same
behaviour. The only change is that they're now an explicit opt-in via
get_encoding_mappings() instead of being applied unconditionally to every
dataset regardless of domain.
"""

from typing import Dict, List, Tuple

BINARY_MAPPINGS: Dict[str, Dict[str, int]] = {
    "FamilyHistory": {"Yes": 1, "No": 0},
    "Recurrence":    {"Yes": 1, "No": 0},
    "Gender":        {"Female": 0, "Male": 1},
}

ORDINAL_MAPPINGS: Dict[str, Dict[str, int]] = {
    "SmokingStatus": {"Non-Smoker": 0, "Former Smoker": 1, "Smoker": 2},
    "Stage": {"I": 0, "II": 1, "III": 2, "IV": 3},
    "TreatmentResponse": {
        "No Response": 0, "Partial Remission": 1, "Complete Remission": 2
    },
    "Survival_Category": {
        "Very_Short": 0, "Short": 1, "Medium": 2, "Long": 3, "Very_Long": 4
    },
    "Age_Group": {"Young": 0, "Middle_Age": 1, "Senior": 2, "Elderly": 3},
    "BMI_Category": {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3},
    "Tumor_Size_Category": {"Small": 0, "Medium": 1, "Large": 2},
    "Cancer_Stage": {"I": 0, "II": 1, "III": 2, "IV": 3},
    "Physical_Activity": {"Low": 0, "Moderate": 1, "High": 2},
    "Diet_Risk": {"Low": 0, "Medium": 1, "High": 2},
}

NOMINAL_COLUMNS: List[str] = [
    "Race/Ethnicity", "CancerType", "TreatmentType",
    "HospitalRegion", "Country", "GeneticMarker",
]

ID_COLUMNS: List[str] = ["PatientID", "Patient_ID", "patient_id"]


def get_encoding_mappings() -> Tuple[Dict[str, Dict], Dict[str, Dict], List[str]]:
    """
    Returns (binary_mappings, ordinal_mappings, nominal_columns) ready to pass
    to ml_framework.preprocessing.encoding.encode_dataframe().
    """
    return dict(BINARY_MAPPINGS), dict(ORDINAL_MAPPINGS), list(NOMINAL_COLUMNS)
