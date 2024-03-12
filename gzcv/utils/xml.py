from typing import Dict

import cv2


def parse_xml(xml_path: str) -> Dict:
    """
    Summary: parse xml file to dict format
    Args:
        xml_path: path to camera calibration file (i.e., **/ETH-XGaze/calibration/cam_calibration/cam**.xml)
    Returns:
        parse_res: dictionary of camera matrices or etc. [str, (np.ndarry of float)]
    """
    data = cv2.FileStorage(xml_path, cv2.FileStorage_READ)
    keys = data.root().keys()

    parse_res = {}
    for key in keys:
        node = data.getNode(key)
        try:
            item = node.mat()
        except:
            item = node.real()
        parse_res[key] = item
    return parse_res
