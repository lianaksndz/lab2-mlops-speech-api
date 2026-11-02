from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ProjectStructureTests(unittest.TestCase):

    def test_api_file_exists(self):
        self.assertTrue((ROOT / "src/api.py").is_file())

    def test_model_file_exists(self):
        self.assertTrue((ROOT / "src/model.py").is_file())

    def test_training_file_exists(self):
        self.assertTrue((ROOT / "src/train.py").is_file())

    def test_dockerfiles_exist(self):
        self.assertTrue((ROOT / "Dockerfile.single").is_file())
        self.assertTrue((ROOT / "Dockerfile.multistage").is_file())

    def test_api_endpoints_are_defined(self):
        api = (ROOT / "src/api.py").read_text()
        self.assertIn("/health", api)
        self.assertIn("/model-info", api)
        self.assertIn("/predict", api)

    def test_multistage_runtime_excludes_training_code(self):
        dockerfile = (ROOT / "Dockerfile.multistage").read_text()
        self.assertIn(
            "COPY src/model.py src/api.py ./src/",
            dockerfile
        )


if __name__ == "__main__":
    unittest.main()
