"""Smoke tests for all SENTINEL model modules.

Verifies that each model:
  1. Imports without error
  2. Initializes with default parameters
  3. Produces output with correct shapes on a forward pass
  4. Computes loss without error (where applicable)

Run with:
    conda run -n physiformer pytest tests/test_models.py -v
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BATCH_SIZE = 4
EMBED_DIM = 256


# ===== Phase 3: Biology =====

class TestSpeciesHealth:
    def test_import(self):
        from sentinel.models.biology.species_health import SentinelSpeciesHealthIndex
        assert SentinelSpeciesHealthIndex is not None

    def test_forward(self):
        from sentinel.models.biology.species_health import SentinelSpeciesHealthIndex
        model = SentinelSpeciesHealthIndex()
        emb = torch.randn(BATCH_SIZE, EMBED_DIM)
        covs = torch.randn(BATCH_SIZE, 5)
        out = model(emb, site_covariates=covs)
        assert out.health_scores.shape == (BATCH_SIZE, 6)
        assert out.occupancy_probs.shape == (BATCH_SIZE, 6)
        assert out.stressor_logits.shape[0] == BATCH_SIZE

    def test_loss(self):
        from sentinel.models.biology.species_health import SentinelSpeciesHealthIndex
        model = SentinelSpeciesHealthIndex()
        emb = torch.randn(BATCH_SIZE, EMBED_DIM)
        out = model(emb)
        targets = {
            "health_scores": torch.rand(BATCH_SIZE, 6) * 100,
            "occupancy": torch.randint(0, 2, (BATCH_SIZE, 6)).float(),
        }
        loss, per_task = model.compute_loss(out, targets)
        assert loss.requires_grad
        assert not torch.isnan(loss)


class TestDiseaseForecaster:
    def test_import(self):
        from sentinel.models.biology.disease_forecast import IntegratedDiseaseRisk
        assert IntegratedDiseaseRisk is not None

    def test_forward(self):
        from sentinel.models.biology.disease_forecast import IntegratedDiseaseRisk
        model = IntegratedDiseaseRisk()
        emb = torch.randn(BATCH_SIZE, EMBED_DIM)
        doy = torch.randint(1, 366, (BATCH_SIZE,))
        out = model(emb, doy)
        assert out.cyanotoxin is not None
        assert out.vibrio is not None
        assert out.naegleria is not None
        assert out.schistosomiasis is not None
        assert out.alert_level.shape == (BATCH_SIZE,)


class TestARGSurveillance:
    def test_import(self):
        from sentinel.models.biology.arg_surveillance import ARGPredictor
        assert ARGPredictor is not None

    def test_forward_otu(self):
        from sentinel.models.biology.arg_surveillance import ARGPredictor
        model = ARGPredictor()
        x = torch.randn(BATCH_SIZE, 5000)
        out = model(x, input_type="otu")
        assert out.log_abundance.shape == (BATCH_SIZE, 8)
        assert out.burden_score.shape == (BATCH_SIZE,)

    def test_forward_embedding(self):
        from sentinel.models.biology.arg_surveillance import ARGPredictor
        model = ARGPredictor()
        x = torch.randn(BATCH_SIZE, EMBED_DIM)
        out = model(x, input_type="embedding")
        assert out.log_abundance.shape == (BATCH_SIZE, 8)


# ===== Phase 4: Digital Twin =====

class TestDigitalTwin:
    def test_import(self):
        from sentinel.models.twin.twin_engine import DigitalTwinEngine
        assert DigitalTwinEngine is not None

    def test_forward(self):
        from sentinel.models.twin.twin_engine import DigitalTwinEngine
        model = DigitalTwinEngine()
        emb = torch.randn(BATCH_SIZE, EMBED_DIM)
        out = model(emb, horizons=(1, 7, 14))
        assert out.predictions.shape[1] == BATCH_SIZE
        assert out.predictions.shape[2] == 10

    def test_physics_only(self):
        from sentinel.models.twin.twin_engine import DigitalTwinEngine
        model = DigitalTwinEngine()
        emb = torch.randn(BATCH_SIZE, EMBED_DIM)
        traj = model.physics_only_forward(emb, horizons=(1, 7))
        assert traj.shape[1] == BATCH_SIZE


class TestCounterfactual:
    def test_import(self):
        from sentinel.models.twin.counterfactual import CounterfactualSimulator
        assert CounterfactualSimulator is not None


class TestRestoration:
    def test_import(self):
        from sentinel.models.twin.restoration import RestorationOutcomePredictor
        assert RestorationOutcomePredictor is not None


class TestClimateCoupling:
    def test_import(self):
        from sentinel.models.twin.climate_coupling import (
            ClimateEncoder, ClimateModulator, SeasonalPrior,
        )
        assert ClimateEncoder is not None

    def test_climate_encoder(self):
        from sentinel.models.twin.climate_coupling import ClimateEncoder
        enc = ClimateEncoder()
        x = torch.randn(BATCH_SIZE, 30, 8)  # 30 days, 8 vars
        doy = torch.arange(180, 210).unsqueeze(0).expand(BATCH_SIZE, -1)
        out = enc(x, doy)
        assert out.shape == (BATCH_SIZE, 30, 128)

    def test_seasonal_prior(self):
        from sentinel.models.twin.climate_coupling import SeasonalPrior
        sp = SeasonalPrior()
        doy = torch.tensor([90.0, 180.0, 270.0, 360.0])
        out = sp(doy)
        assert out.shape == (4, 10)


# ===== Phase 1: Fusion =====

class TestMoMEFusion:
    def test_import(self):
        from sentinel.models.fusion.mome import MoMEFusion
        assert MoMEFusion is not None


class TestStreamGNN:
    def test_import(self):
        from sentinel.models.graph.stream_gnn import StreamEncoder
        assert StreamEncoder is not None


# ===== Phase 4 (continued): Recovery & Bioremediation =====

class TestRecoveryPlanner:
    def test_import(self):
        from sentinel.models.twin.recovery_planner import RecoveryPlanner
        assert RecoveryPlanner is not None

    def test_forward(self):
        from sentinel.models.twin.recovery_planner import RecoveryPlanner
        model = RecoveryPlanner()
        emb = torch.randn(BATCH_SIZE, EMBED_DIM)
        species_idx = torch.zeros(BATCH_SIZE, dtype=torch.long)
        out = model(emb, species_idx)
        assert out is not None


class TestBioremediation:
    def test_import(self):
        from sentinel.models.biology.bioremediation import BioremediationRecommender
        assert BioremediationRecommender is not None

    def test_forward(self):
        from sentinel.models.biology.bioremediation import BioremediationRecommender
        model = BioremediationRecommender()
        emb = torch.randn(BATCH_SIZE, EMBED_DIM)
        out = model(emb)
        assert out is not None


# ===== Phase 3 (continued): Occupancy, FDR, AOP, Metatranscriptomic =====

class TestOccupancy:
    def test_import(self):
        from sentinel.models.biology.occupancy import OccupancyShiftModel, eDNACommunityPredictor
        assert OccupancyShiftModel is not None
        assert eDNACommunityPredictor is not None

    def test_occupancy_shift(self):
        from sentinel.models.biology.occupancy import OccupancyShiftModel
        model = OccupancyShiftModel()
        emb = torch.randn(BATCH_SIZE, EMBED_DIM)
        out = model(emb)
        assert out.delta_occupancy.shape[0] == BATCH_SIZE
        assert out.baseline_occupancy.shape[0] == BATCH_SIZE

    def test_edna_community(self):
        from sentinel.models.biology.occupancy import eDNACommunityPredictor
        model = eDNACommunityPredictor()
        sensor_emb = torch.randn(BATCH_SIZE, EMBED_DIM)
        hydro_emb = torch.randn(BATCH_SIZE, EMBED_DIM)
        out = model(sensor_emb, hydro_emb)
        assert out.otu_presence_prob is not None
        assert out.otu_log_abundance is not None


class TestFieldDoseResponse:
    def test_import(self):
        from sentinel.models.biology.field_dose_response import FieldDoseResponseModel
        assert FieldDoseResponseModel is not None

    def test_forward(self):
        from sentinel.models.biology.field_dose_response import FieldDoseResponseModel
        model = FieldDoseResponseModel()
        emb = torch.randn(BATCH_SIZE, EMBED_DIM)
        conc = torch.randn(BATCH_SIZE, 24)
        out = model(emb, conc)
        assert out.implied_exposure is not None
        assert out.risk_levels is not None
        assert out.margin_of_safety is not None


class TestInverseAOP:
    def test_import(self):
        from sentinel.models.biology.inverse_aop import InverseAOPPredictor
        assert InverseAOPPredictor is not None

    def test_forward(self):
        from sentinel.models.biology.inverse_aop import InverseAOPPredictor
        model = InverseAOPPredictor()
        emb = torch.randn(BATCH_SIZE, EMBED_DIM)
        out = model(emb)
        assert out.activation_prob is not None
        assert out.severity_score is not None
        assert out.alert_levels is not None


class TestMetatranscriptomic:
    def test_import(self):
        from sentinel.models.biology.metatranscriptomic import MetatranscriptomicSurveillance
        assert MetatranscriptomicSurveillance is not None

    def test_forward(self):
        from sentinel.models.biology.metatranscriptomic import MetatranscriptomicSurveillance
        model = MetatranscriptomicSurveillance()
        gene_features = torch.randn(BATCH_SIZE, 10000)
        sentinel_emb = torch.randn(BATCH_SIZE, EMBED_DIM)
        out = model(gene_features, sentinel_emb)
        assert out.detection_scores is not None
        assert out.log_abundance is not None
        assert out.alert_levels is not None


# ===== Phase 5: Platform =====

class TestEJOverlay:
    def test_import(self):
        from sentinel.platform.ej_overlay import EJOverlayEngine
        assert EJOverlayEngine is not None


class TesteDNAKit:
    def test_import(self):
        from sentinel.platform.edna_kit import eDNAValidator, eDNAIngestion
        assert eDNAValidator is not None
        assert eDNAIngestion is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
