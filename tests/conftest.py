import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Make the repository packages, an in-place build of the extension and the test
# helpers importable without installation.
for path in (ROOT, os.path.join(ROOT, "smi_torch", "csrc"), os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)


def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="CUDA is not available")
        for item in items:
            item.add_marker(skip)
