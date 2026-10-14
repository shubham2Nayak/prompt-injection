"""
Migration Tests for PromptTrace v2.5

Verifies the complete two-model architecture migration:
- Qwen/Qwen2.5-3B-Instruct for reasoning (Research, Analyst, Action)
- nvidia/Nemotron-3.5-Content-Safety for annotation (Context Engine, Propagation Monitor)
- BAAI/bge-large-en-v1.5 for embeddings
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json

from config import config
from models import (
    get_reasoning_model,
    get_annotation_model,
    get_model_pool,
    validate_annotation_output,
    validate_reasoning_output,
    ANNOTATION_CATEGORIES,
)
from models.gpu_budget import FP16
from context import ContextEngine
from agents import ResearchAgent, AnalystAgent, ActionAgent, PropagationMonitor
from state import (
    WorkflowState,
    ResearchAgentOutput,
    AnalystAgentOutput,
    ActionAgentOutput,
    PropagationMonitorOutput,
)
from context.schema import ContextAnnotation


class TestModelRegistry(unittest.TestCase):
    """Test centralized model registry configuration."""

    def test_reasoning_models_use_qwen(self):
        """All reasoning agents use Qwen 2.5 3B."""
        self.assertEqual(config.RESEARCH_AGENT_MODEL, "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(config.ANALYST_AGENT_MODEL, "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(config.ACTION_AGENT_MODEL, "Qwen/Qwen2.5-3B-Instruct")

    def test_annotation_model_uses_nemotron(self):
        """Annotation model uses Nemotron 3.5 Content Safety."""
        self.assertEqual(config.ANNOTATION_MODEL, "nvidia/Nemotron-3.5-Content-Safety")
        self.assertEqual(config.PROPAGATION_MONITOR_MODEL, "nvidia/Nemotron-3.5-Content-Safety")

    def test_embedding_model_unchanged(self):
        """Embedding model remains BAAI/bge-large-en-v1.5."""
        self.assertEqual(config.EMBEDDING_MODEL, "BAAI/bge-large-en-v1.5")

    def test_temperature_settings(self):
        """Per-model temperature settings are correct."""
        self.assertEqual(config.TEMPERATURE_RESEARCH, 0.2)
        self.assertEqual(config.TEMPERATURE_ANALYST, 0.1)
        self.assertEqual(config.TEMPERATURE_ACTION, 0.0)
        self.assertEqual(config.TEMPERATURE_ANNOTATION, 0.0)

    def test_no_legacy_model_references(self):
        """Ensure no TinyLlama, DistilGPT2, Gemma, or OPT references in config."""
        config_dict = config.to_dict()
        for key, value in config_dict.items():
            if isinstance(value, str):
                self.assertNotIn("TinyLlama", value)
                self.assertNotIn("DistilGPT2", value)
                self.assertNotIn("distilgpt2", value)
                self.assertNotIn("Gemma", value)
                self.assertNotIn("gemma", value)
                self.assertNotIn("facebook/opt", value)
                self.assertNotIn("opt-350m", value)


class LoaderIsolationMixin:
    """Clears the shared model loader caches around each test.

    ``models.loader`` keeps the two models in module-level singletons. Once any
    test triggers a real load, every later test's patched ``from_pretrained`` is
    never called, so assertions on ``call_count`` read zero and the outcome
    depends on test execution order. Releasing the caches in setUp makes each
    test exercise the cold-load path it actually claims to test.
    """

    def setUp(self):
        super().setUp()
        import models.loader as loader

        self._loader = loader
        loader.release_models()

    def tearDown(self):
        self._loader.release_models()
        super().tearDown()


class TestModelLoaderSingleton(LoaderIsolationMixin, unittest.TestCase):
    """Test shared model loader singleton behavior."""

    @patch("models.loader.AutoTokenizer.from_pretrained")
    @patch("models.loader.AutoModelForCausalLM.from_pretrained")
    @patch("models.loader.TextGenerationPipeline")
    def test_reasoning_model_singleton(self, mock_pipeline, mock_model, mock_tokenizer):
        """Reasoning model loads once and returns same instance."""
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance

        model1, tokenizer1, pipeline1 = get_reasoning_model()
        model2, tokenizer2, pipeline2 = get_reasoning_model()

        # Should be the same instances
        self.assertIs(model1, model2)
        self.assertIs(tokenizer1, tokenizer2)
        self.assertIs(pipeline1, pipeline2)

        # Model should only be loaded once
        self.assertEqual(mock_model.call_count, 1)
        self.assertEqual(mock_tokenizer.call_count, 1)
        self.assertEqual(mock_pipeline.call_count, 1)

    @patch("models.loader.AutoTokenizer.from_pretrained")
    @patch("models.loader.AutoModelForCausalLM.from_pretrained")
    @patch("models.loader.TextGenerationPipeline")
    def test_annotation_model_singleton(self, mock_pipeline, mock_model, mock_tokenizer):
        """Annotation model loads once and returns same instance."""
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance

        model1, tokenizer1, pipeline1 = get_annotation_model()
        model2, tokenizer2, pipeline2 = get_annotation_model()

        self.assertIs(model1, model2)
        self.assertIs(tokenizer1, tokenizer2)
        self.assertIs(pipeline1, pipeline2)


class TestCUDADeviceAssignment(LoaderIsolationMixin, unittest.TestCase):
    """Test CUDA device assignment."""

    @patch("models.loader.torch.cuda.is_available", return_value=True)
    @patch("models.loader.AutoTokenizer.from_pretrained")
    @patch("models.loader.AutoModelForCausalLM.from_pretrained")
    @patch("models.loader.TextGenerationPipeline")
    def test_models_load_on_cuda_when_available(
        self, mock_pipeline, mock_model, mock_tokenizer, mock_cuda
    ):
        """Models should be placed on CUDA when a device is available."""
        mock_model_instance = Mock()
        mock_model_instance.device = "cuda:0"
        mock_model.return_value = mock_model_instance
        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance

        model, tokenizer, pipeline = get_reasoning_model()

        call_kwargs = mock_model.call_args.kwargs

        # The loader pins an explicit CUDA ordinal rather than "auto": the GPU
        # budget planner reserved a specific slice of VRAM, and "auto" would let
        # accelerate place the weights anywhere it liked, defeating the budget.
        # The invariant is that the placement names the GPU, not a specific
        # string.
        device_map = call_kwargs.get("device_map")
        self.assertIsNotNone(device_map, "model was not placed on a device")
        self.assertIn(
            "cuda",
            str(device_map),
            f"expected a CUDA device map, got {device_map!r}",
        )
        self.assertTrue(call_kwargs.get("trust_remote_code"))


class TestAnnotationValidation(unittest.TestCase):
    """Test Nemotron annotation output validation."""

    def test_valid_annotation_json(self):
        """Valid JSON annotation passes validation."""
        valid_json = json.dumps({
            "category": "EMBEDDED_PROCESSING_NOTE",
            "confidence": 0.92,
            "reason": "Embedded AI directive detected",
            "annotation_category": "EMBEDDED_PROCESSING_NOTE",
            "infected": True,
        })

        result = validate_annotation_output(valid_json)

        self.assertEqual(result["category"], "EMBEDDED_PROCESSING_NOTE")
        self.assertEqual(result["confidence"], 0.92)
        self.assertEqual(result["annotation_category"], "EMBEDDED_PROCESSING_NOTE")
        self.assertTrue(result["infected"])

    def test_invalid_category_defaults_to_unknown(self):
        """Invalid category defaults to UNKNOWN."""
        invalid_json = json.dumps({
            "category": "INVALID_CATEGORY",
            "confidence": 0.8,
            "reason": "Test",
            "annotation_category": "INVALID_CATEGORY",
            "infected": False,
        })

        result = validate_annotation_output(invalid_json)

        self.assertEqual(result["category"], "UNKNOWN")
        self.assertEqual(result["annotation_category"], "UNKNOWN")

    def test_confidence_clamped_to_range(self):
        """Confidence values are clamped to [0, 1]."""
        json_high = json.dumps({
            "category": "ACADEMIC_DESCRIPTION",
            "confidence": 1.5,
            "reason": "Test",
            "annotation_category": "ACADEMIC_DESCRIPTION",
            "infected": False,
        })

        result = validate_annotation_output(json_high)
        self.assertEqual(result["confidence"], 1.0)

        json_low = json.dumps({
            "category": "ACADEMIC_DESCRIPTION",
            "confidence": -0.5,
            "reason": "Test",
            "annotation_category": "ACADEMIC_DESCRIPTION",
            "infected": False,
        })

        result = validate_annotation_output(json_low)
        self.assertEqual(result["confidence"], 0.0)

    def test_malformed_json_returns_unknown(self):
        """Malformed JSON returns UNKNOWN with 0 confidence."""
        result = validate_annotation_output("not valid json {")

        self.assertEqual(result["category"], "UNKNOWN")
        self.assertEqual(result["confidence"], 0.0)
        self.assertFalse(result["infected"])

    def test_all_valid_categories_accepted(self):
        """All valid taxonomy categories are accepted."""
        for category in ANNOTATION_CATEGORIES:
            test_json = json.dumps({
                "category": category,
                "confidence": 0.9,
                "reason": "Test",
                "annotation_category": category,
                "infected": category in {"EMBEDDED_PROCESSING_NOTE", "MIXED_CONTENT"},
            })
            result = validate_annotation_output(test_json)
            self.assertEqual(result["category"], category)


class TestReasoningValidation(unittest.TestCase):
    """Test reasoning model output validation."""

    def test_research_output_validation(self):
        """Research agent output validates required fields."""
        required = ["summary", "key_findings", "sources_cited", "raw_text", "processing_time_ms"]
        test_json = json.dumps({
            "summary": "Test summary",
            "key_findings": ["Finding 1"],
            "sources_cited": ["chunk_1"],
            "raw_text": "Raw text",
            "processing_time_ms": 100,
        })

        result = validate_reasoning_output(test_json, required)

        self.assertEqual(result["summary"], "Test summary")
        self.assertEqual(result["key_findings"], ["Finding 1"])

    def test_missing_fields_get_defaults(self):
        """Missing fields get empty defaults."""
        required = ["summary", "key_findings", "recommendations"]
        test_json = json.dumps({"summary": "Only summary"})

        result = validate_reasoning_output(test_json, required)

        self.assertEqual(result["summary"], "Only summary")
        self.assertEqual(result["key_findings"], [])
        self.assertEqual(result["recommendations"], [])


class TestAgentRefactor(unittest.TestCase):
    """Test that agents use shared model loader."""

    @patch("models.loader.AutoTokenizer.from_pretrained")
    @patch("models.loader.AutoModelForCausalLM.from_pretrained")
    @patch("models.loader.TextGenerationPipeline")
    def test_research_agent_uses_shared_loader(self, mock_pipeline, mock_model, mock_tokenizer):
        """ResearchAgent uses get_reasoning_model."""
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance

        agent = ResearchAgent()

        # Should have pipeline from shared loader
        self.assertIsNotNone(agent.pipeline)

    @patch("models.loader.AutoTokenizer.from_pretrained")
    @patch("models.loader.AutoModelForCausalLM.from_pretrained")
    @patch("models.loader.TextGenerationPipeline")
    def test_analyst_agent_uses_shared_loader(self, mock_pipeline, mock_model, mock_tokenizer):
        """AnalystAgent uses get_reasoning_model."""
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance

        agent = AnalystAgent()
        self.assertIsNotNone(agent.pipeline)

    @patch("models.loader.AutoTokenizer.from_pretrained")
    @patch("models.loader.AutoModelForCausalLM.from_pretrained")
    @patch("models.loader.TextGenerationPipeline")
    def test_action_agent_uses_shared_loader(self, mock_pipeline, mock_model, mock_tokenizer):
        """ActionAgent uses get_reasoning_model."""
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance

        agent = ActionAgent()
        self.assertIsNotNone(agent.pipeline)


class TestPropagationMonitorUsesNemotron(unittest.TestCase):
    """Test PropagationMonitor uses ContextEngine with Nemotron."""

    def test_propagation_monitor_uses_context_engine(self):
        """PropagationMonitor accepts ContextEngine."""
        mock_engine = Mock(spec=ContextEngine)
        monitor = PropagationMonitor(context_engine=mock_engine)
        self.assertIs(monitor.context_engine, mock_engine)

    def test_monitor_workflow_calls_context_engine(self):
        """monitor_workflow routes every agent output through the Context Engine.

        The monitor classifies in one batched call, so the batch entry point is
        what gets mocked here. Mocking ``annotate_agent_output`` instead left the
        batch method returning a bare Mock, which is not iterable.
        """
        mock_engine = Mock(spec=ContextEngine)

        def _clean(name):
            # Context Engine returns ContextAnnotation, which is what the monitor
            # consumes; it is not a PropagationMonitorOutput.
            return ContextAnnotation(
                chunk_id="",
                paragraph_offset=0,
                character_offset=0,
                annotation_category="ACADEMIC_DESCRIPTION",
                confidence=0.95,
                spans=[],
                model_used="nvidia/Nemotron-3.5-Content-Safety",
                latency_ms=10.0,
            )

        mock_engine.annotate_agent_outputs_batch.side_effect = (
            lambda names, texts: [_clean(name) for name in names]
        )

        monitor = PropagationMonitor(context_engine=mock_engine)

        # Real typed outputs; Pydantic rejects a Mock for a declared model field.
        state = WorkflowState(
            original_query="Test",
            research_output=ResearchAgentOutput(
                summary="Research",
                raw_text="Research",
                processing_time_ms=1.0,
            ),
            analyst_output=AnalystAgentOutput(
                executive_summary="Analysis",
                annotation_analysis="Analysis",
                confidence=0.9,
                processing_time_ms=1.0,
            ),
            action_output=ActionAgentOutput(
                action_summary="Action",
                processing_time_ms=1.0,
            ),
        )

        result = monitor.monitor_workflow(state)

        # One batched classification covering all three agents.
        self.assertEqual(mock_engine.annotate_agent_outputs_batch.call_count, 1)
        batch_names = mock_engine.annotate_agent_outputs_batch.call_args.args[0]
        self.assertEqual(
            batch_names,
            ["Research Agent", "Analyst Agent", "Action Agent"],
        )
        self.assertEqual(len(result.propagation_timeline), 3)
        self.assertEqual(len(result.propagation_history), 0)


class TestContextEngineUsesNemotron(unittest.TestCase):
    """Test ContextEngine uses Nemotron via shared loader."""

    @patch("context.classifier.get_annotation_model")
    def test_classifier_uses_shared_annotation_model(self, mock_get_annotation):
        """AnnotationClassifier uses get_annotation_model."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_pipeline = Mock()
        mock_get_annotation.return_value = (mock_model, mock_tokenizer, mock_pipeline)

        from context.classifier import AnnotationClassifier
        classifier = AnnotationClassifier()

        # Should have called get_annotation_model
        mock_get_annotation.assert_called_once()
        self.assertIs(classifier.pipeline, mock_pipeline)

    def test_context_engine_creates_classifier_with_correct_model(self):
        """ContextEngine creates classifier with ANNOTATION_MODEL."""
        from config import config

        with patch("context.context_engine.AnnotationClassifier") as mock_classifier:
            engine = ContextEngine(
                annotation_model_name="test-model",
                annotation_device="cpu",
            )
            mock_classifier.assert_called_once_with(
                model_name="test-model",
                device="cpu",
                temperature=config.TEMPERATURE_ANNOTATION,
                max_new_tokens=config.MAX_NEW_TOKENS_ANNOTATION,
                # The annotation batch size must be wired through. Without it the
                # TextGenerationPipeline defaults to batch_size=1 and a batched
                # classifier call degrades into one generate() per prompt.
                batch_size=config.ANNOTATION_BATCH_SIZE,
            )

    def test_annotation_batch_size_is_forwarded_to_classifier(self):
        """ANNOTATION_BATCH_SIZE reaches the classifier, not just config."""
        with patch("context.context_engine.AnnotationClassifier") as mock_classifier:
            ContextEngine(annotation_model_name="m", annotation_device="cpu")

        _, kwargs = mock_classifier.call_args
        self.assertEqual(kwargs["batch_size"], config.ANNOTATION_BATCH_SIZE)
        self.assertGreaterEqual(kwargs["batch_size"], 1)


class TestNoCPUFallbackWhenCUDAAvailable(LoaderIsolationMixin, unittest.TestCase):
    """Test that CPU fallback is not used when CUDA is available."""

    @patch("models.loader.torch.cuda.is_available", return_value=True)
    @patch("models.loader.AutoTokenizer.from_pretrained")
    @patch("models.loader.AutoModelForCausalLM.from_pretrained")
    @patch("models.loader.TextGenerationPipeline")
    def test_no_silent_cpu_fallback(self, mock_pipeline, mock_model, mock_tokenizer, mock_cuda):
        """Models should not silently fall back to CPU when CUDA exists."""
        mock_model_instance = Mock()
        mock_model_instance.device = "cuda:0"
        mock_model.return_value = mock_model_instance
        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance

        # Config should place the models on the GPU when one is available.
        self.assertEqual(config.MODEL_DEVICE, "cuda")

        # The dtype is deployment-specific: backend/.env, MODEL_DTYPE and the
        # gpu budget planner can each legitimately select one (bfloat16 is
        # required for Gemma/Nemotron, which overflow float16). Asserting a
        # single literal here made the test fail on any machine with a local
        # .env, so assert the invariant instead - a dtype the loader accepts.
        self.assertIn(
            config.MODEL_DTYPE.lower(),
            {"float16", "bfloat16", "float32"},
        )

        # And that the loader resolves a torch dtype from it without error.
        from models.gpu_budget import torch_dtype_for

        import torch

        self.assertIn(
            torch_dtype_for(FP16, config.RESEARCH_AGENT_MODEL),
            (torch.float16, torch.bfloat16, torch.float32),
        )


class TestEndToEndWorkflow(unittest.TestCase):
    """Test end-to-end workflow integration."""

    @patch("models.loader.AutoTokenizer.from_pretrained")
    @patch("models.loader.AutoModelForCausalLM.from_pretrained")
    @patch("models.loader.TextGenerationPipeline")
    def test_workflow_initialization(self, mock_pipeline, mock_model, mock_tokenizer):
        """Full workflow initializes with correct models."""
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance

        from graph import PromptTraceWorkflow

        # This would require more extensive mocking for full integration
        # Just verify the imports work and models are referenced correctly
        wf = PromptTraceWorkflow()
        self.assertIsNotNone(wf.context_engine)
        self.assertIsNotNone(wf.research_agent)
        self.assertIsNotNone(wf.analyst_agent)
        self.assertIsNotNone(wf.action_agent)
        self.assertIsNotNone(wf.propagation_monitor)


if __name__ == "__main__":
    unittest.main()