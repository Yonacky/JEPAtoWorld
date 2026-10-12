import sys
from pathlib import Path
from unittest.mock import patch
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core/e03"))
from device_utils import select_device

class DevicePolicyTests(unittest.TestCase):
    def test_no_cuda_falls_back(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(str(select_device("auto")), "cpu")
    def test_most_free_visible_cuda(self):
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=2), patch("torch.empty"), patch("torch.cuda.mem_get_info", side_effect=[(10, 100), (70, 100)]):
            self.assertEqual(str(select_device("auto")), "cuda:1")
    def test_unusable_cuda_skipped(self):
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=2), patch("torch.empty", side_effect=[RuntimeError("unavailable"), object()]), patch("torch.cuda.mem_get_info", return_value=(20, 100)):
            self.assertEqual(str(select_device("auto")), "cuda:1")
    def test_explicit_cuda_does_not_silently_fallback(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError): select_device("cuda:0")
    def test_explicit_cpu(self):
        self.assertEqual(str(select_device("cpu")), "cpu")
    def test_mps_float64_not_supported(self):
        with self.assertRaises(ValueError): select_device("mps")

if __name__ == "__main__": unittest.main()
