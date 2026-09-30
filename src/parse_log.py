"""train.py 학습 로그(stdout)를 ndarray로 파싱."""

import re
import sys

import numpy as np

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s*\|\s*Loss:\s*([\d.]+)\s*\|\s*Val (?:F1|Acc):\s*([\d.]+)\s*\|\s*Best:\s*([\d.]+)"
)
TEST_RE = re.compile(r"Final Test (?:F1|Accuracy):\s*([\d.]+)")


def parse_log(text):
    """
    반환값:
        epochs: ndarray[N, 4]  각 행 = (epoch, loss, val_metric, best_so_far)
        test_metric: float | None  마지막 "Final Test ..." 값
    """
    rows = [
        (int(epoch), float(loss), float(val), float(best))
        for epoch, loss, val, best in EPOCH_RE.findall(text)
    ]
    epochs = np.array(rows, dtype=np.float64)

    m = TEST_RE.search(text)
    test_metric = float(m.group(1)) if m else None

    return epochs, test_metric


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    text = open(path, encoding="utf-8").read() if path else sys.stdin.read()

    epochs, test_metric = parse_log(text)
    print(f"epochs shape: {epochs.shape}  (columns: epoch, loss, val_metric, best)")
    print(epochs)
    print(f"test_metric: {test_metric}")
