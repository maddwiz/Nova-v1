"""Tests for G6: Learned Slot Prediction."""
from usc.predict import SlotPredictor, SlotModel


def _training_data():
    """Repetitive slot values — IPs, status codes, timestamps."""
    return [
        ["10.0.0.1", "200", "GET", "/api/v1/users"],
        ["10.0.0.1", "200", "GET", "/api/v1/users"],
        ["10.0.0.1", "200", "GET", "/api/v1/items"],
        ["10.0.0.2", "200", "POST", "/api/v1/users"],
        ["10.0.0.1", "404", "GET", "/api/v1/missing"],
        ["10.0.0.1", "200", "GET", "/api/v1/users"],
        ["10.0.0.1", "200", "GET", "/api/v1/users"],
        ["10.0.0.3", "200", "GET", "/api/v1/health"],
        ["10.0.0.1", "200", "GET", "/api/v1/users"],
        ["10.0.0.1", "500", "GET", "/api/v1/crash"],
    ]


class TestTraining:
    def test_train_returns_model(self):
        p = SlotPredictor()
        model = p.train(_training_data())
        assert isinstance(model, SlotModel)

    def test_model_has_slot_freqs(self):
        p = SlotPredictor()
        model = p.train(_training_data())
        assert 0 in model.slot_freqs  # IP slot
        assert 1 in model.slot_freqs  # status code slot

    def test_top_value_is_most_frequent(self):
        p = SlotPredictor()
        model = p.train(_training_data())
        # "10.0.0.1" appears 8/10 times in slot 0
        assert model.top_value(0) == "10.0.0.1"
        # "200" appears 8/10 times in slot 1
        assert model.top_value(1) == "200"
        # "GET" appears 9/10 times in slot 2
        assert model.top_value(2) == "GET"

    def test_prediction_accuracy(self):
        p = SlotPredictor()
        model = p.train(_training_data())
        # "10.0.0.1" is 8/10 = 80%
        assert model.prediction_accuracy(0) == 0.8
        # "200" is 8/10 = 80%
        assert model.prediction_accuracy(1) == 0.8

    def test_unknown_slot_returns_none(self):
        p = SlotPredictor()
        model = p.train(_training_data())
        assert model.top_value(99) is None
        assert model.prediction_accuracy(99) == 0.0


class TestPrediction:
    def test_predict_returns_top_value(self):
        p = SlotPredictor()
        p.train(_training_data())
        assert p.predict(0) == "10.0.0.1"
        assert p.predict(2) == "GET"

    def test_predict_without_training_returns_none(self):
        p = SlotPredictor()
        assert p.predict(0) is None


class TestEncoding:
    def test_encode_decode_roundtrip_all_predicted(self):
        p = SlotPredictor()
        model = p.train(_training_data())
        # Values that match predictions exactly
        values = ["10.0.0.1", "200", "GET", "/api/v1/users"]
        encoded = p.encode_with_prediction(values, model)
        decoded, _ = p.decode_with_prediction(encoded, model)
        assert decoded == values

    def test_encode_decode_roundtrip_none_predicted(self):
        p = SlotPredictor()
        model = p.train(_training_data())
        # Values that don't match predictions
        values = ["10.0.0.99", "503", "DELETE", "/api/v2/new"]
        encoded = p.encode_with_prediction(values, model)
        decoded, _ = p.decode_with_prediction(encoded, model)
        assert decoded == values

    def test_encode_decode_mixed(self):
        p = SlotPredictor()
        model = p.train(_training_data())
        # Mix of predicted and unpredicted
        values = ["10.0.0.1", "503", "GET", "/api/v2/new"]
        encoded = p.encode_with_prediction(values, model)
        decoded, _ = p.decode_with_prediction(encoded, model)
        assert decoded == values

    def test_predicted_values_save_space(self):
        p = SlotPredictor()
        model = p.train(_training_data())
        # All predicted
        predicted_vals = ["10.0.0.1", "200", "GET", "/api/v1/users"]
        enc_predicted = p.encode_with_prediction(predicted_vals, model)
        # None predicted
        novel_vals = ["10.0.0.99", "503", "DELETE", "/api/v2/new"]
        enc_novel = p.encode_with_prediction(novel_vals, model)
        # Predicted should be smaller (1 byte per slot vs full string)
        assert len(enc_predicted) < len(enc_novel)

    def test_empty_values(self):
        p = SlotPredictor()
        model = p.train(_training_data())
        encoded = p.encode_with_prediction([], model)
        decoded, _ = p.decode_with_prediction(encoded, model)
        assert decoded == []


class TestCompressionRatio:
    def test_high_repetition_compresses_well(self):
        p = SlotPredictor()
        data = _training_data()
        model = p.train(data)
        ratio = p.compression_ratio(data, model)
        # Highly repetitive data should compress below 1.0
        assert ratio < 1.0

    def test_unique_data_expands(self):
        p = SlotPredictor()
        # Train on data A, encode data B (all misses)
        model = p.train(_training_data())
        novel = [[f"unique_{i}_{j}" for j in range(4)] for i in range(10)]
        ratio = p.compression_ratio(novel, model)
        # All misses → overhead from flags → ratio > 1.0
        assert ratio > 1.0
