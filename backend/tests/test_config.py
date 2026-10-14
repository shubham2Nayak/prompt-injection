import unittest
from unittest.mock import patch

import config


class ConfigMemoryFallbackTests(unittest.TestCase):
    def test_default_device_returns_cuda_when_available(self):
        with patch.object(config.torch, "cuda", create=True) as mock_cuda:
            mock_cuda.is_available.return_value = True
            mock_cuda.mem_get_info.return_value = (8 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024)

            self.assertEqual(config._default_device(), "cuda")

    def test_default_device_returns_cpu_when_cuda_unavailable(self):
        with patch.object(config.torch, "cuda", create=True) as mock_cuda:
            mock_cuda.is_available.return_value = False

            self.assertEqual(config._default_device(), "cpu")

    def test_default_device_returns_cpu_when_torch_unavailable(self):
        with patch.object(config, "torch", None):
            self.assertEqual(config._default_device(), "cpu")


class ConfigModelRegistryTests(unittest.TestCase):
    def test_reasoning_model_defaults(self):
        self.assertEqual(config.RESEARCH_MODEL, "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(config.ANALYST_MODEL, "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(config.ACTION_MODEL, "Qwen/Qwen2.5-3B-Instruct")

    def test_annotation_model_default(self):
        self.assertEqual(config.ANNOTATION_MODEL, "nvidia/Nemotron-3.5-Content-Safety")

    def test_embedding_model_default(self):
        self.assertEqual(config.EMBEDDING_MODEL, "BAAI/bge-large-en-v1.5")


if __name__ == "__main__":
    unittest.main()