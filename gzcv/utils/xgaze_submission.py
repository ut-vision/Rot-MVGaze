import os
import shutil
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt


def format_and_save_as_xgaze_submission(
    preds: List[Dict[str, Any]], gaze_key: str, save_dir: str
) -> None:
    """
    Summary: Make ETH-XGaze submission file.
             The structure of the submission file is
             |- submission_within_eva.zip -> submission_within_eva/
              ` submission_within_eva_results.txt
    """
    pred_gazes: List[npt.NDArray] = []
    for pred in preds:
        pred_gaze_batch: npt.NDArray = pred[gaze_key]
        pred_gazes.extend(pred_gaze_batch.cpu().numpy().tolist())

    zip_dir = os.path.join(save_dir, "submission_within_eva")
    if not os.path.exists(zip_dir):
        os.mkdir(zip_dir)
    np.savetxt(
        os.path.join(zip_dir, "within_eva_results.txt"),
        pred_gazes,
        delimiter=",",
    )
    shutil.make_archive(
        zip_dir, format="zip", root_dir=save_dir, base_dir="submission_within_eva"
    )
