"""
Unit tests for model architectures (smoke tests – no GPU required).
"""

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def dummy_batch():
    """Return a small (2, 3, 64, 64) float32 tensor."""
    return torch.rand(2, 3, 64, 64)


class TestEfficientNetICH:
    def test_output_shape(self, dummy_batch):
        from src.models.architectures.efficientnet import build_efficientnet

        model = build_efficientnet(
            model_name="efficientnet_b0",
            num_classes=6,
            pretrained=False,
        )
        model.eval()
        with torch.inference_mode():
            logits = model(dummy_batch)
        assert logits.shape == (2, 6)

    def test_no_nan_in_output(self, dummy_batch):
        from src.models.architectures.efficientnet import build_efficientnet

        model = build_efficientnet(
            model_name="efficientnet_b0",
            num_classes=6,
            pretrained=False,
        )
        model.eval()
        with torch.inference_mode():
            logits = model(dummy_batch)
        assert not torch.isnan(logits).any()


class TestResNetICH:
    def test_output_shape(self, dummy_batch):
        from src.models.architectures.resnet import build_resnet

        model = build_resnet(
            model_name="resnet18",
            num_classes=6,
            pretrained=False,
        )
        model.eval()
        with torch.inference_mode():
            logits = model(dummy_batch)
        assert logits.shape == (2, 6)


class TestFocalLoss:
    def test_loss_positive(self):
        from src.models.train import FocalLoss

        criterion = FocalLoss()
        logits = torch.randn(4, 6)
        targets = torch.randint(0, 2, (4, 6)).float()
        loss = criterion(logits, targets)
        assert loss.item() > 0

    def test_perfect_prediction_low_loss(self):
        from src.models.train import FocalLoss

        criterion = FocalLoss(gamma=2.0)
        # Very confident correct predictions
        logits = torch.full((4, 6), 10.0)
        targets = torch.ones(4, 6)
        loss = criterion(logits, targets)
        assert loss.item() < 0.01
