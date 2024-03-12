import re


class DictRegex:
    def __init__(self, pattern):
        self.pattern = re.compile(pattern)

    def match(self, data):
        matched = {}
        for key, item in data.items():
            res = self.pattern.fullmatch(key)
            if res is not None:
                matched[key] = item
        return matched

    def match_and_extract(self, data):
        matched = {}
        for key, item in data.items():
            res = self.pattern.findall(key)
            if len(res) > 0:
                matched[key] = (res[0], item)
        return matched


if __name__ == "__main__":
    dict_example = {
        "pred_gaze_0": 3,
        "pred_gaze_1": 4,
        "pred_gaze_aux_0": 1,
        "pred_gaze": 5,
        "gt_gaze_0": 10,
        "gt_gaze_1": 13,
    }
    pattern = "pred_gaze(.*)"
    # pattern = "pred_gaze.*"
    dict_regex = DictRegex(pattern)
    result = dict_regex.match(dict_example)
    print(result)
    result_extracted = dict_regex.match_and_extract(dict_example)
    print(result_extracted)
