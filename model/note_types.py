
from typing import Dict

notes_label_to_id: Dict[str, int] = {
    "no_hit": 0,
    "don": 1,
    "ka": 2,
    "bigDon": 3,
    "bigKa": 4,
    "drumroll": 5,
    "bigDrumroll": 6,
    "balloon": 7,
}

notes_id_to_label = {v : k for v, k in notes_label_to_id.items()}