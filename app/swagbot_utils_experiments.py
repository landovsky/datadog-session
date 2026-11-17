#!/usr/bin/env python3

# Suppress Google Cloud warnings at the very beginning
import os
os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
os.environ.setdefault('GLOG_minloglevel', '3')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

"""
SwagBot LangGraph Model Comparison Experiments

This module provides comprehensive model comparison functionality for the SwagBot LangGraph workflow,
enabling systematic testing and evaluation of different LLM models using Datadog LLM Observability.

Platform-Aware: Automatically detects LLM_PLATFORM environment variable and tests appropriate models.

Supported Platforms:
- AWS Bedrock: Claude variants (Haiku, Sonnet, 3.5-Sonnet, etc.)
- Google Vertex AI: Gemini models (Flash, Pro, Ultra)
- OpenAI: GPT models (4o-mini, 4o, 4-turbo)
- Azure OpenAI: GPT models (4o-mini, 4o, 4)

Usage:
    # Bedrock (LLM_PLATFORM=bedrock)
    python swagbot_utils_experiments.py --model-comparison claude-3-haiku
    python swagbot_utils_experiments.py --model-comparison claude-3-sonnet
    
    # Vertex AI (LLM_PLATFORM=vertex)
    python swagbot_utils_experiments.py --model-comparison gemini-flash
    python swagbot_utils_experiments.py --model-comparison gemini-pro
    
    # OpenAI (LLM_PLATFORM=openai)
    python swagbot_utils_experiments.py --model-comparison gpt-4o-mini
    python swagbot_utils_experiments.py --model-comparison gpt-4o
    
    # Universal commands (work with any platform)
    python swagbot_utils_experiments.py --compare-all-models
    python swagbot_utils_experiments.py --list-models
"""

import logging
import argparse
from typing import Dict, List, Any
from datetime import datetime

# Suppress Google Cloud warnings BEFORE importing any Google libraries
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GOOGLE_CLOUD_DISABLE_GRPC_LOGS'] = '1'

# Suppress absl logging at the C++ level
import sys
if hasattr(sys, 'stderr'):
    # Redirect stderr temporarily to suppress C++ warnings
    import io
    original_stderr = sys.stderr
    sys.stderr = io.StringIO()

# Datadog LLM Observability for Experiments
from ddtrace.llmobs import LLMObs, RemoteEvaluator

# Import your existing workflow
from swagbot_langgraph_workflow import process_swagbot_request
from swagbot_langgraph_config import LangChainConfig

# Restore stderr after imports
if 'original_stderr' in locals():
    sys.stderr = original_stderr

# Set up minimal logging
logging.basicConfig(level=logging.ERROR)  # Only show errors by default
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Keep info level just for this module

# Reduce verbosity of other modules
logging.getLogger('swagbot_langgraph_config').setLevel(logging.ERROR)
logging.getLogger('swagbot_langgraph_workflow').setLevel(logging.ERROR)
logging.getLogger('botocore').setLevel(logging.ERROR)
logging.getLogger('langchain_google_vertexai').setLevel(logging.ERROR)
logging.getLogger('google').setLevel(logging.ERROR)

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message="All log messages before absl::InitializeLog()")
warnings.filterwarnings("ignore", message="ALTS creds ignored")
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress Google Cloud and ALTS warnings at environment level
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

# Suppress absl logging warnings
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    absl.logging.set_stderrthreshold(absl.logging.ERROR)
    # Disable absl logging completely
    absl.logging.use_python_logging()
except ImportError:
    # absl not installed, no action needed
    pass

# Additional suppression for Google Cloud libraries
try:
    import google.cloud.logging
    google.cloud.logging.Client().setup_logging(log_level=logging.ERROR)
except ImportError:
    # Google Cloud logging not available, no action needed
    pass

# Suppress gRPC warnings at the Python level
try:
    import grpc
    # Monkey patch gRPC logging
    original_grpc_log = grpc._common._LOGGER
    grpc._common._LOGGER = logging.getLogger('grpc').setLevel(logging.ERROR)
except (ImportError, AttributeError):
    # gRPC not available or internal structure changed, no action needed
    pass

# Manual LangGraph instrumentation patch (required for ddtrace 3.12+)
try:
    from ddtrace import patch
    patch(langgraph=True)
    logger.debug("Manual LangGraph instrumentation patch applied")
except Exception as e:
    logger.warning(f"️ Failed to apply LangGraph patch in experiments: {e}")

class SuppressStderr:
    """Context manager to suppress stderr output during Google Cloud operations."""
    
    def __init__(self):
        """Initialize the context manager."""
        self.original_stderr = None
        self.devnull = None
    
    def __enter__(self):
        import sys
        import os
        self.original_stderr = sys.stderr
        self.devnull = open(os.devnull, 'w')
        sys.stderr = self.devnull
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import sys
        if self.devnull:
            self.devnull.close()
        sys.stderr = self.original_stderr

class SwagBotExperimentBase:
    """Dataset creation and evaluation methods shared across experiment suites."""

    def __init__(self):
        pass

    # ===============================================================================
    # DATASET CREATION
    # ===============================================================================

    def create_customer_service_dataset(self) -> Any:
        """Create or update dataset for Customer Service agent evaluation."""
        records = [
            {
                "input_data": "I bought a mug but it arrived broken, can I get a replacement?",
                "expected_output": "We apologize for the damaged mug. We'll process a replacement immediately. Our return policy covers damaged items with free replacement shipping."
            },
            {
                "input_data": "What's your return policy for items bought on sale?",
                "expected_output": "Sale items can be returned within 30 days for store credit or exchange. Original packaging required. Final sale items (marked as such) cannot be returned."
            },
            {
                "input_data": "I can't track my order, it's been 5 days since I placed it",
                "expected_output": "I'll help you track your order. Orders typically ship within 2-3 business days. Please provide your order number so I can check the current status and provide tracking information."
            },
            {
                "input_data": "The headphones I received don't match the description online",
                "expected_output": "I'm sorry the headphones don't match expectations. We'll arrange a return and full refund. Please keep the original packaging and we'll email you a prepaid return label."
            },
            {
                "input_data": "I was charged twice for the same order, can you help?",
                "expected_output": "Double charges are usually pre-authorizations that will be released. I'll check your account and if there's an actual duplicate charge, we'll process an immediate refund within 3-5 business days."
            }
        ]
        return LLMObs.create_dataset(
            dataset_name="swagbot_customer_service_eval",
            description="Customer service scenarios for SwagBot evaluation",
            records=records
        )

    def create_product_specialist_dataset(self) -> Any:
        """Create or update dataset for Product Specialist agent evaluation."""
        records = [
            {
                "input_data": "Tell me about the Dog Steel Bottle",
                "expected_output": "The Dog Steel Bottle is a robust stainless steel vacuum flask with sports lid, priced at $29.99. It's perfect for keeping drinks hot or cold, made from durable materials."
            },
            {
                "input_data": "How much do the Dog Headphones cost?",
                "expected_output": "The Dog Headphones are priced at $229.99. They feature soft, plush cushions for excellent comfort and sound isolation."
            },
            {
                "input_data": "Do you have any eco-friendly products?",
                "expected_output": "Our stainless steel products like the Dog Steel Bottle are eco-friendly - reusable, durable, and made from recyclable materials that reduce single-use plastic waste."
            },
            {
                "input_data": "What categories of products do you sell?",
                "expected_output": "We offer products in several categories including accessories (headphones, bottles), apparel (t-shirts), drinkware (mugs), and lifestyle items."
            }
        ]
        return LLMObs.create_dataset(
            dataset_name="swagbot_product_specialist_eval",
            description="Product information scenarios for SwagBot evaluation",
            records=records
        )

    def create_product_specialist_benchmark_dataset(self) -> Any:
        """Create dataset seeded with the same low-quality scenarios used in Lab 4.

        These records mirror the 3 annotation-queue traces that students export in Lab 4
        plus one passing example captured after the Lab 3 fixes.  They are used as a
        fallback when the annotation queue is empty or hasn't been processed yet.
        """
        records = [
            {
                "input_data": "What headphones do you recommend for under $250?",
                "expected_output": (
                    "The Dog Headphones at $229.99 are a great fit for your $250 budget. "
                    "They feature soft, plush cushions that seal you in for an immersive "
                    "listening experience. Note: only 3 units left in stock — order soon."
                ),
            },
            {
                "input_data": "Tell me about the Dog Steel Bottle",
                "expected_output": (
                    "The Dog Steel Bottle (NoLimit) is a robust stainless steel vacuum flask "
                    "with a sports lid, priced at $29.99. Currently in stock with 42 units available."
                ),
            },
            {
                "input_data": "What is the difference between your Dog Steel Bottle and Dog Plastic Bottle?",
                "expected_output": (
                    "Here's a comparison of our two bottle options:\n\n"
                    "Dog Steel Bottle — $29.99. Robust stainless steel vacuum flask with sports lid. "
                    "In stock (42 units). Best for temperature retention.\n\n"
                    "Dog Plastic Bottle — $15.99. Lightweight and durable for daily hydration. "
                    "In stock (55 units). Best for a light, budget-friendly option."
                ),
            },
            {
                "input_data": "How much is the Dog Steel Bottle?",
                "expected_output": (
                    "The Dog Steel Bottle (NoLimit) is priced at $29.99. "
                    "It's a robust stainless steel vacuum flask with a sports lid. "
                    "Currently in stock with 42 units available."
                ),
            },
        ]
        return LLMObs.create_dataset(
            dataset_name="product_specialist_benchmark",
            description="Low-quality Product Specialist traces from annotation queue (Lab 4 ground truth)",
            records=records,
        )

    def create_promotion_specialist_dataset(self) -> Any:
        """Create dataset for Promotion Specialist agent evaluation."""
        records = [
            {
                "input_data": "Are there any current discounts for the steel bottle?",
                "expected_output": "Yes! Use code STEELBOTTLE10 for 10% off the Dog Steel Bottle. This promotion is valid until December 31st, 2024."
            },
            {
                "input_data": "Do you have any deals on t-shirts this week?",
                "expected_output": "Currently we have TSHIRT15 for 15% off all t-shirts, and BUNDLE20 for 20% off when you buy 2 or more t-shirts."
            },
            {
                "input_data": "What promotions are running right now?",
                "expected_output": "Active promotions include: STEELBOTTLE10 (10% off steel bottles), TSHIRT15 (15% off t-shirts), BUNDLE20 (20% off 2+ t-shirts), and FREESHIP50 (free shipping on orders over $50)."
            },
            {
                "input_data": "Is there a discount code for first-time buyers?",
                "expected_output": "Welcome! New customers can use WELCOME20 for 20% off their first order over $30. This offer is valid for 30 days from signup."
            },
            {
                "input_data": "Any coupon codes for the holidays?",
                "expected_output": "Holiday specials include HOLIDAY25 for 25% off orders over $75, and GIFT15 for 15% off gift items. Both valid through December 31st."
            }
        ]
        return LLMObs.create_dataset(
            dataset_name="swagbot_promotion_specialist_eval",
            description="Promotion and discount scenarios for SwagBot evaluation",
            records=records
        )

    def create_comprehensive_dataset(self) -> Any:
        """Create or update comprehensive dataset covering multiple agent scenarios."""
        records = [
            {
                "input_data": "Tell me about the Dog Steel Bottle and any current promotions",
                "expected_output": "The Dog Steel Bottle ($24.99) is a robust stainless steel vacuum flask with sports lid. Currently, you can save 10% with code STEELBOTTLE10, making it just $22.49!"
            },
            {
                "input_data": "I want to buy headphones but want to see customer reviews first",
                "expected_output": "Our Dog Headphones ($229.99) have excellent reviews for comfort and sound quality. Customers love the soft, plush cushions. I can help you find detailed reviews and ratings."
            },
            {
                "input_data": "What's your return policy and current deals on bottles?",
                "expected_output": "Returns are accepted within 30 days with original packaging. For bottles, we currently have STEELBOTTLE10 for 10% off the Dog Steel Bottle, making it $22.49 instead of $24.99."
            },
            {
                "input_data": "Show me your best selling products and any deals available",
                "expected_output": "Our bestsellers include Dog Headphones ($229.99) and Dog Steel Bottle ($24.99). Current deals: STEELBOTTLE10 (10% off bottles), TSHIRT15 (15% off shirts), and FREESHIP50 (free shipping over $50)."
            },
            {
                "input_data": "I need help with a damaged product and want to know about your eco-friendly options",
                "expected_output": "For damaged products, we offer free replacements with prepaid return shipping. Our eco-friendly options include the reusable Dog Steel Bottle made from recyclable stainless steel."
            }
        ]
        return LLMObs.create_dataset(
            dataset_name="swagbot_comprehensive_eval",
            description="Multi-agent scenarios testing complete SwagBot workflow",
            records=records
        )

    # ===============================================================================
    # EVALUATOR FUNCTIONS
    # ===============================================================================

    def exact_match_evaluator(self, input_data: str, output_data: str, expected_output: str) -> bool:  # noqa: ARG002
        """Simple exact match evaluator."""
        return output_data.strip().lower() == expected_output.strip().lower()

    def contains_key_info_evaluator(self, input_data: str, output_data: str, expected_output: str) -> bool:  # noqa: ARG002
        """Boolean evaluator that checks if output contains key information from expected output."""
        import re

        unavailable_phrases = ["price not available", "not available", "price information not available"]
        expected_lower = expected_output.lower()
        output_lower = output_data.lower()

        if any(phrase in expected_lower for phrase in unavailable_phrases):
            unavailable_indicators = [
                "not available", "no price", "price not", "don't have price",
                "cannot provide price", "price information is not", "pricing is not available",
                "price is not listed", "no pricing information", "unavailable"
            ]
            return any(indicator in output_lower for indicator in unavailable_indicators)

        key_terms = []
        prices = re.findall(r'\$[\d,]+\.?\d*', expected_output)
        key_terms.extend(prices)

        products = ['Dog Steel Bottle', 'Dog Headphones', 'headphones', 'bottle', 'steel bottle']
        for product in products:
            if product.lower() in expected_lower:
                key_terms.append(product)

        promo_codes = re.findall(r'[A-Z]+\d+', expected_output)
        key_terms.extend(promo_codes)

        if not key_terms:
            return True

        found_terms = sum(1 for term in key_terms if term.lower() in output_lower)
        return (found_terms / len(key_terms)) >= 0.9

    def response_quality_evaluator(self, input_data: str, output_data: str, expected_output: str) -> float:
        """Score-based comprehensive response quality evaluator (0.0-1.0)."""

        output_lower = output_data.lower()
        input_lower = input_data.lower()
        score_components = []

        # 1. Length Quality (0.2 weight)
        response_length = len(output_data.strip())
        if response_length == 0:
            length_score = 0.0
        elif response_length < 20:
            length_score = 0.3
        elif response_length < 100:
            length_score = 0.7
        elif response_length < 500:
            length_score = 1.0
        elif response_length < 1000:
            length_score = 0.8
        else:
            length_score = 0.5
        score_components.append((length_score, 0.2))

        # 2. Key Information Match (0.4 weight)
        key_info_score = self.contains_key_info_evaluator(input_data, output_data, expected_output)
        score_components.append((key_info_score, 0.4))

        # 3. Response Appropriateness (0.2 weight)
        appropriateness_score = 1.0
        negative_phrases = ['error', 'sorry, i cannot', "i don't know", 'i am unable', 'i cannot help']
        if any(phrase in output_lower for phrase in negative_phrases):
            appropriateness_score -= 0.5
        if not any(punct in output_data for punct in ['.', '!', '?']):
            appropriateness_score -= 0.3
        if output_data.strip() == '' or len(output_data.strip().split()) < 5:
            appropriateness_score = 0.0
        score_components.append((max(0.0, appropriateness_score), 0.2))

        # 4. Context Relevance (0.2 weight)
        relevance_score = 0.7
        if any(word in input_lower for word in ['price', 'cost', 'how much']):
            relevance_score = 1.0 if any(i in output_lower for i in ['$', 'price', 'cost', 'not available']) else 0.3
        if any(word in input_lower for word in ['about', 'tell me', 'information', 'describe']):
            if any(word in output_lower for word in ['product', 'description', 'features', 'specifications']):
                relevance_score = 1.0
        if any(word in input_lower for word in ['discount', 'deal', 'promotion', 'coupon']):
            relevance_score = 1.0 if any(w in output_lower for w in ['discount', 'code', '%', 'promotion', 'deal']) else 0.3
        score_components.append((relevance_score, 0.2))

        final_score = sum(score * weight for score, weight in score_components)
        return round(max(0.0, min(1.0, final_score)), 2)

    def output_similarity_evaluator(self, input_data: str, output_data: str, expected_output: str) -> float:  # noqa: ARG002
        """F1 word overlap between output and expected_output (0.0–1.0). No external dependencies."""
        if not expected_output or not output_data:
            return 0.0
        # Dataset records from annotation queue exports may arrive as dicts — normalize to str
        if not isinstance(expected_output, str):
            expected_output = str(expected_output)
        if not isinstance(output_data, str):
            output_data = str(output_data)
        stopwords = {"a", "an", "the", "is", "it", "in", "of", "to", "and", "or", "for", "with", "that", "this", "i", "you", "we"}
        output_tokens = set(w.lower().strip(".,!?") for w in output_data.split() if w.lower() not in stopwords)
        expected_tokens = set(w.lower().strip(".,!?") for w in expected_output.split() if w.lower() not in stopwords)
        if not output_tokens or not expected_tokens:
            return 0.0
        intersection = output_tokens & expected_tokens
        precision = len(intersection) / len(output_tokens)
        recall = len(intersection) / len(expected_tokens)
        if precision + recall == 0:
            return 0.0
        return round(2 * precision * recall / (precision + recall), 3)

    def html_format_evaluator(self, input_data: str, output_data: str, expected_output: str) -> bool:  # noqa: ARG002
        """Evaluator to check if output contains proper HTML formatting."""
        html_indicators = ['<p>' in output_data, '</p>' in output_data]
        if any(word in input_data.lower() for word in ['price', 'product', 'information']):
            html_indicators.append('<strong>' in output_data)
        if any(word in input_data.lower() for word in ['list', 'products', 'categories', 'options']):
            html_indicators.append('<ul>' in output_data or '<li>' in output_data)
        return sum(html_indicators) >= 2

    # ===============================================================================
    # SUMMARY EVALUATORS
    # ===============================================================================

    def avg_output_similarity_summary(self, inputs, outputs, expected_outputs, evaluators_results):  # noqa: ARG002
        """Summary evaluator: average F1 word-overlap similarity across all records."""
        scores = evaluators_results.get("output_similarity_evaluator", [])
        valid = [s for s in scores if s is not None]
        if not valid:
            return None
        return round(sum(valid) / len(valid), 3)

    def avg_response_quality_summary(self, inputs, outputs, expected_outputs, evaluators_results):  # noqa: ARG002
        """Summary evaluator to calculate average response quality across all experiments."""
        quality_scores = evaluators_results.get("response_quality_evaluator", [])
        valid_scores = [s for s in quality_scores if s is not None]
        if not valid_scores:
            return None
        return round(sum(valid_scores) / len(valid_scores), 3)

    def key_info_match_rate_summary(self, inputs, outputs, expected_outputs, evaluators_results):  # noqa: ARG002
        """Summary evaluator to calculate percentage of outputs containing key information."""
        matches = evaluators_results.get("contains_key_info_evaluator", [])
        if not matches:
            return None
        return round(matches.count(True) / len(matches) * 100, 1)

    def html_format_compliance_summary(self, inputs, outputs, expected_outputs, evaluators_results):  # noqa: ARG002
        """Summary evaluator to calculate percentage of outputs with proper HTML formatting."""
        html_checks = evaluators_results.get("html_format_evaluator", [])
        if not html_checks:
            return None
        return round(html_checks.count(True) / len(html_checks) * 100, 1)

    # ===============================================================================
    # REMOTE EVALUATOR SUMMARIES — aggregates scores from Datadog-configured evaluators
    # ===============================================================================

    def avg_ecommerce_quality_summary(self, inputs, outputs, expected_outputs, evaluators_results):  # noqa: ARG002
        """Summary evaluator: average ecommerce-quality-evaluation score across all records."""
        scores = evaluators_results.get("ecommerce-quality-evaluation", [])
        valid = [s for s in scores if s is not None]
        if not valid:
            return None
        return round(sum(valid) / len(valid), 3)

    def avg_goal_completeness_summary(self, inputs, outputs, expected_outputs, evaluators_results):  # noqa: ARG002
        """Summary evaluator: share of records rated "Excellent - All Goals Complete".

        goal-completeness is a categorical evaluator (string labels), so it is
        aggregated as a rate rather than averaged.
        """
        labels = evaluators_results.get("goal-completeness", [])
        valid = [str(label).strip().lower() for label in labels if label is not None]
        if not valid:
            return None
        return round(sum(1 for label in valid if label.startswith("excellent")) / len(valid), 3)

class SwagBotModelComparisonSuite(SwagBotExperimentBase):
    """
    Platform-aware model comparison experiment suite for SwagBot LangGraph workflow.
    
    Automatically adapts to the configured LLM platform (Bedrock, Vertex AI, OpenAI, Azure).
    
    Features:
    - Multi-platform support (Bedrock, Vertex AI, OpenAI, Azure OpenAI)
    - Platform-specific model selection and configuration
    - Model-specific configuration overrides
    - Comparative evaluation across identical datasets
    - Performance and cost analysis
    - Datadog LLM Observability integration with consistent span naming
    """
    
    # Experiment execution configuration
    EXPERIMENT_PARALLEL_JOBS = 3  # Number of parallel jobs for experiment execution
    
    def __init__(self):
        self.config = LangChainConfig()
        self.platform = self.config.llm_platform
        self.available_models = self._get_available_models_for_platform()
        self._ensure_llmobs_enabled()
        platform_name = {"vertex": "Gemini Enterprise", "bedrock": "AWS Bedrock", "openai": "OpenAI", "azure": "Azure OpenAI"}.get(self.platform, self.platform.upper())
        print(f"Experiments based on LangGraph workflow.")
        print(f"Initializing for {platform_name}.")
        print(f"Models loaded.")
        # RemoteEvaluator references the Datadog-configured evaluators from Lab 2 by name
        self.ecommerce_evaluator = RemoteEvaluator(eval_name="ecommerce-quality-evaluation")
        self.goal_completeness_evaluator = RemoteEvaluator(eval_name="goal-completeness")

    def _get_available_models_for_platform(self) -> Dict[str, Dict[str, Any]]:
        """Get available models based on the current platform."""
        if self.platform == "bedrock":
            return self._get_available_bedrock_models()
        elif self.platform == "vertex":
            return self._get_available_vertex_models()
        elif self.platform == "openai":
            return self._get_available_openai_models()
        elif self.platform == "azure":
            return self._get_available_azure_models()
        else:
            raise ValueError(f"Unsupported platform for experiments: {self.platform}")
    
    def _get_available_bedrock_models(self) -> Dict[str, Dict[str, Any]]:
        """Define the core models for experimentation."""
        return {
            # Claude Models
            "claude-3-haiku": {
                "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
                "display_name": "Claude 3 Haiku",
                "description": "Fast, lightweight model (baseline)",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "low",
                "provider": "anthropic"
            },
            "claude-3-sonnet": {
                "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
                "display_name": "Claude 3 Sonnet",
                "description": "Balanced model for general tasks",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "high",
                "provider": "anthropic"
            },
            "claude-3.5-sonnet": {
                "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "display_name": "Claude 3.5 Sonnet",
                "description": "Latest balanced model with improved capabilities",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "high",
                "provider": "anthropic"
            },
            # Claude 3.7 Sonnet Model
            "claude-3-7-sonnet": {
                "model_id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                "display_name": "Claude 3.7 Sonnet",
                "description": "Latest Claude Sonnet model with improved capabilities",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "high",
                "provider": "anthropic"
            },
            # Claude Sonnet 4 Model
            "claude-sonnet-4": {
                "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
                "provider": "anthropic",
                "display_name": "Claude Sonnet 4",
                "description": "Latest Claude Sonnet model with improved capabilities",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "high"
            }
        }
    
    def _get_available_vertex_models(self) -> Dict[str, Dict[str, Any]]:
        """Define Vertex AI Gemini models for experimentation (location=global required).
        Confirmed available in Instruqt GCP projects as of 2026-10: Gemini 3.5 Flash/Flash Lite + Gemini 3 previews.
        Note: gemini-2.5-flash-lite, gemini-2.5-pro and gemini-3.1-flash-lite-preview return 404.
        max_tokens is high because thinking tokens count against it (1000 truncates answers)."""
        return {
            "gemini-3.5-flash-lite": {
                "model_id": "gemini-3.5-flash-lite",
                "display_name": "Gemini 3.5 Flash Lite",
                "description": "Ultra-fast, lowest-cost model — baseline for comparison",
                "temperature": 0.3,
                "max_tokens": 8192,
                "cost_tier": "low",
                "provider": "google",
                "location": "global"
            },
            "gemini-3.5-flash": {
                "model_id": "gemini-3.5-flash",
                "display_name": "Gemini 3.5 Flash",
                "description": "Fast and efficient with strong instruction following",
                "temperature": 0.3,
                "max_tokens": 8192,
                "cost_tier": "medium",
                "provider": "google",
                "location": "global"
            },
            "gemini-3-flash-preview": {
                "model_id": "gemini-3-flash-preview",
                "display_name": "Gemini 3 Flash Preview",
                "description": "Previous-gen flash preview — speed/quality balance",
                "temperature": 0.3,
                "max_tokens": 8192,
                "cost_tier": "medium",
                "provider": "google",
                "location": "global"
            },
            "gemini-4-argon": {
                "model_id": "gemini-4-argon",
                "display_name": "Gemini 4 Argon",
                "description": "Latest flagship model — frontier capability, highest cost",
                "temperature": 0.3,
                "max_tokens": 8192,
                "cost_tier": "high",
                "provider": "google",
                "location": "global"
            },
            "gemini-3.1-pro-preview": {
                "model_id": "gemini-3.1-pro-preview",
                "display_name": "Gemini 3.1 Pro Preview",
                "description": "Pro model — highest capability available",
                "temperature": 0.3,
                "max_tokens": 8192,
                "cost_tier": "high",
                "provider": "google",
                "location": "global"
            },
        }
    
    def _get_available_openai_models(self) -> Dict[str, Dict[str, Any]]:
        """Define OpenAI models for experimentation."""
        return {
            "gpt-4o-mini": {
                "model_id": "gpt-4o-mini",
                "display_name": "GPT-4o Mini",
                "description": "Fast, cost-effective model for quick responses",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "low",
                "provider": "openai"
            },
            "gpt-4o": {
                "model_id": "gpt-4o",
                "display_name": "GPT-4o",
                "description": "Advanced model for complex reasoning tasks",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "medium",
                "provider": "openai"
            },
            "gpt-4-turbo": {
                "model_id": "gpt-4-turbo",
                "display_name": "GPT-4 Turbo",
                "description": "High-performance model for demanding tasks",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "high",
                "provider": "openai"
            }
        }
    
    def _get_available_azure_models(self) -> Dict[str, Dict[str, Any]]:
        """Define Azure OpenAI models for experimentation."""
        return {
            "gpt-4o-mini": {
                "model_id": "gpt-4o-mini",
                "display_name": "GPT-4o Mini (Azure)",
                "description": "Fast, cost-effective model for quick responses",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "low",
                "provider": "azure-openai"
            },
            "gpt-4o": {
                "model_id": "gpt-4o",
                "display_name": "GPT-4o (Azure)",
                "description": "Advanced model for complex reasoning tasks",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "medium",
                "provider": "azure-openai"
            },
            "gpt-4": {
                "model_id": "gpt-4",
                "display_name": "GPT-4 (Azure)",
                "description": "High-performance model for demanding tasks",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "high",
                "provider": "azure-openai"
            }
        }
    
    def _ensure_llmobs_enabled(self):
        """Enable Datadog LLM Observability (prerequisite)."""
        # Check for experiment-specific app key
        experiments_app_key = os.getenv('SWAGBOT_DD_APP_KEY')
        api_key = os.getenv('DD_API_KEY')
        
        if not experiments_app_key:
            logger.warning("No SWAGBOT_DD_APP_KEY found - experiments may have limited functionality")
        if not api_key:
            logger.warning("No DD_API_KEY found - experiments may not work properly")
        
        # Enable LLMObs with consistent naming [[memory:3279953]]
        LLMObs.enable(
            site=os.getenv('DD_SITE', 'datadoghq.com'),
            api_key=api_key,
            app_key=experiments_app_key,  # Use dedicated app key for experiments
            project_name="swagbot_optimization"  # Project name for experiments
        )
        
        logger.debug("Datadog LLM Observability enabled for experiments")
    

    # ===============================================================================
    # MODEL-SPECIFIC TASK WRAPPER
    # ===============================================================================
    
    def create_direct_agent_task(self, model_key: str, agent_type: str = "product_specialist"):
        """Create a task function that tests a specific agent directly without the full workflow."""

        def direct_agent_task(input_data: str, config: Dict[str, Any]) -> str:  # noqa: ARG001
            """
            Direct agent testing that bypasses the full workflow.
            Tests only the specified agent with the given model, using the same
            RAG pipeline and prompt as the production workflow.
            """
            try:
                from swagbot_langgraph_workflow import (
                    _rag_search_raw, _format_docs_for_llm, _resources_dir, LLMFactory,
                )
                from swagbot_langgraph_config import LangChainConfig
                from swagbot_utils import SynthesisUtils
                from langchain_core.messages import SystemMessage, HumanMessage

                model_config = self.available_models[model_key]
                user_query = input_data if isinstance(input_data, str) else str(input_data)

                # Load the same prompt as the production workflow
                agent_prompt_key = agent_type.replace("_", "-")
                resources_dir = _resources_dir()
                try:
                    prompt_text = SynthesisUtils.load_agent_prompt(agent_prompt_key, resources_dir)
                except Exception as prompt_err:
                    logger.warning(f"Could not load prompt for {agent_prompt_key} from {resources_dir}: {prompt_err}")
                    prompt_text = f"You are a SwagBot {agent_type.replace('_', ' ')} specialist. Help the customer thoroughly."

                # RAG retrieval — same pipeline as production
                docs = _rag_search_raw([agent_type], user_query)
                logger.debug(f"RAG retrieved {len(docs)} docs for agent={agent_type} resources={resources_dir}")

                # Build messages — same pattern as production llm_node
                # Always use the same template structure so prompt tracking sees a single template fingerprint
                rag_context = _format_docs_for_llm(docs) if docs else "No relevant products found in knowledge base."
                human_content = (
                    f"Context from knowledge base:\n\n{rag_context}"
                    f"\n\nCustomer question: {user_query}"
                )
                messages = [SystemMessage(content=prompt_text), HumanMessage(content=human_content)]

                # Instantiate the model for this specific model_key
                cfg = LangChainConfig()
                model_cfg = {
                    "model_id": model_config["model_id"],
                    "temperature": model_config.get("temperature", 0.3),
                    "max_tokens": model_config.get("max_tokens", 1000),
                    "project_id": cfg.google_project_id,
                    "location": model_config.get("location", cfg.google_location),
                }
                with SuppressStderr():
                    llm = LLMFactory.create(cfg.llm_platform, model_cfg)
                    response = llm.invoke(messages)

                content = response.content
                if isinstance(content, list):
                    # Gemini 3.x returns content as a list of parts (text + thought_signature)
                    content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
                return content if isinstance(content, str) else str(content)

            except Exception as e:
                logger.error(f"Error in direct agent task for {model_key}: {e}")
                model_display = self.available_models.get(model_key, {}).get("display_name", model_key)
                return f"Error testing {model_display} on {agent_type}: {str(e)}"

        return direct_agent_task
    
    def create_model_specific_task(self, model_key: str):
        """Create a task function that uses a specific model configuration for full workflow."""
        
        def model_task(input_data: str, config: Dict[str, Any]) -> str:  # noqa: ARG001
            """
            Model-specific wrapper that overrides the default model configuration.
            
            This function:
            1. Temporarily overrides the model configuration
            2. Uses the existing process_swagbot_request function with new model
            3. Extracts and returns the output
            4. Uses consistent span naming for Datadog [[memory:4152491]]
            """
            try:
                model_config = self.available_models[model_key]
                original_config = {}
                
                # Temporarily override environment variables for this model
                env_overrides = {
                    'SPECIALIST_MODEL': model_config.get('inference_profile', model_config['model_id']),
                    'PLANNING_MODEL': model_config.get('inference_profile', model_config['model_id']),
                    'SYNTHESIZER_MODEL': model_config.get('inference_profile', model_config['model_id'])
                }
                
                # Store original values
                for key, value in env_overrides.items():
                    original_config[key] = os.getenv(key)
                    os.environ[key] = value
                
                try:
                    # Call workflow with new model configuration
                    result = process_swagbot_request(input_data)
                    
                    # Extract the output
                    if isinstance(result, dict):
                        output = result.get('output', str(result))
                    else:
                        output = str(result)
                    
                    # Ensure we return a string
                    return output if isinstance(output, str) else str(output)
                    
                finally:
                    # Restore original environment
                    for key, original_value in original_config.items():
                        if original_value is not None:
                            os.environ[key] = original_value
                        elif key in os.environ:
                            del os.environ[key]
                            
            except Exception as e:
                logger.error(f"Error in model_task for {model_key}: {e}")
                return f"Error processing request with {model_config['display_name']}: {str(e)}"
        
        return model_task
    
    # ===============================================================================
    # MODEL COMPARISON EXPERIMENTS
    # ===============================================================================
    
    def run_single_model_experiment(self, model_key: str, dataset_type: str = "comprehensive", direct_agent: bool = False) -> tuple:
        """Run experiment with a specific model using existing dataset."""
        if model_key not in self.available_models:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(self.available_models.keys())}")
        
        start_time = datetime.now()
        model_config = self.available_models[model_key]

        # Get dataset name — known aliases map to internal names, unknown values are raw dataset names
        dataset_aliases = {
            "customer_service": "swagbot_customer_service_eval",
            "product_specialist": "swagbot_product_specialist_eval",
            "promotion_specialist": "swagbot_promotion_specialist_eval",
            "comprehensive": "swagbot_comprehensive_eval"
        }
        dataset_name = dataset_aliases.get(dataset_type, dataset_type)

        dataset = LLMObs.pull_dataset(dataset_name)
        if dataset is None:
            raise ValueError(f"Dataset '{dataset_name}' not found. Please create it first using --create-dataset {dataset_type}")

        # Annotation queue UI exports store expected_output as a list and metadata as a
        # JSON string. Normalize both before ddtrace unpacks the records.
        for record in getattr(dataset, "_records", []):
            eo = record.get("expected_output")
            if isinstance(eo, list):
                record["expected_output"] = eo[0] if eo else ""
            elif eo is not None and not isinstance(eo, str):
                record["expected_output"] = str(eo)
            meta = record.get("metadata")
            if isinstance(meta, str):
                import json as _json
                try:
                    record["metadata"] = _json.loads(meta)
                except (ValueError, TypeError):
                    record["metadata"] = {}
            elif meta is None:
                record["metadata"] = {}

        # Extract and log dataset version for reproducibility
        dataset_version = getattr(dataset, 'current_version', None) or getattr(dataset, 'version', 'unknown')
        logger.debug("Using dataset '%s' version: %s", dataset_name, dataset_version)
            
        # Log experiment start
        # Get number of records from the dataset
        try:
            if hasattr(dataset, 'records'):
                num_records = len(dataset.records)
            elif isinstance(dataset, dict) and 'records' in dataset:
                num_records = len(dataset['records'])
            else:
                num_records = len(dataset)  # Assume dataset is directly a list of records
            print(f"\n  {model_config['display_name']}  ({num_records} records)")
        except Exception as e:
            logger.warning(f"Could not determine number of records in dataset: {e}")
            print(f"\n  {model_config['display_name']}")
        
        # Create task function based on test mode
        if direct_agent:
            # Direct agent testing - no HTML evaluation needed
            model_task = self.create_direct_agent_task(model_key, "product_specialist")
            evaluators = [
                self.output_similarity_evaluator,
                self.ecommerce_evaluator,           # RemoteEvaluator — ecommerce-quality-evaluation (lab 2)
                self.goal_completeness_evaluator,   # RemoteEvaluator — goal-completeness (lab 2)
            ]
            summary_evaluators = [
                self.avg_output_similarity_summary,
                self.avg_ecommerce_quality_summary,
                self.avg_goal_completeness_summary,
            ]
            experiment_name = f"Product Specialist agent — {model_config['display_name']}"
            description = f"Direct product_specialist agent testing with {model_config['display_name']} on {dataset_type}"
        else:
            # Full workflow testing
            model_task = self.create_model_specific_task(model_key)
            evaluators = [
                self.contains_key_info_evaluator,
                self.response_quality_evaluator,
                self.html_format_evaluator,
                self.ecommerce_evaluator,  # RemoteEvaluator — ecommerce-quality-evaluation (lab 2)
            ]
            summary_evaluators = [
                self.avg_response_quality_summary,
                self.key_info_match_rate_summary,
                self.html_format_compliance_summary,
                self.avg_ecommerce_quality_summary,
            ]
            experiment_name = f"Full workflow testing of {model_config['display_name']} on {dataset_type} scenarios"
            description = experiment_name
        
        experiment = LLMObs.experiment(
            name=experiment_name,
            task=model_task,
            dataset=dataset,
            evaluators=evaluators,
            summary_evaluators=summary_evaluators,
            description=description,
            tags={"model_name": model_key},
            config={"model_name": model_key}
        )
        
        # Run experiment with parallel processing for faster execution
        result = experiment.run(jobs=self.EXPERIMENT_PARALLEL_JOBS)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        url = experiment.url if hasattr(experiment, 'url') else None
        print(f"  Completed in {duration:.1f}s")
        return (result, experiment, duration, url)
    
    def run_model_comparison_suite(self, models_to_test: List[str] = None, dataset_type: str = "comprehensive", direct_agent: bool = False) -> Dict[str, Any]:
        """Run comparison across multiple models using the same dataset."""
        if models_to_test is None:
            # Test all 3 core models
            models_to_test = ["claude-haiku", "claude-instant", "claude-sonnet"]
        
        valid_models = [m for m in models_to_test if m in self.available_models]
        skipped = [m for m in models_to_test if m not in self.available_models]
        if skipped:
            logger.warning("Skipping unknown models: %s", ', '.join(skipped))

        display_names = [self.available_models[m]['display_name'] for m in valid_models]
        print(f"\nComparing models on dataset: {dataset_type}")
        for name in display_names:
            print(f"  - {name}")
        print(f"  {'-'*50}")

        start_time = datetime.now()
        results = {}
        rows = []

        for model_key in valid_models:
            try:
                result, experiment, dur, url = self.run_single_model_experiment(model_key, dataset_type, direct_agent)
                results[model_key] = {
                    "experiment_result": result,
                    "model_config": self.available_models[model_key],
                    "status": "completed"
                }
                rows.append((self.available_models[model_key]['display_name'], f"{dur:.1f}s", "completed", url or ""))
            except Exception as e:
                logger.error(f"Failed to test {model_key}: {e}")
                results[model_key] = {"error": str(e), "model_config": self.available_models[model_key], "status": "failed"}
                rows.append((self.available_models[model_key]['display_name'], "-", "failed", ""))

        total = (datetime.now() - start_time).total_seconds()
        completed = sum(1 for r in results.values() if r["status"] == "completed")

        col_model = max(len(r[0]) for r in rows) + 2
        col_dur = 10
        print(f"\n  {'='*80}")
        print(f"  Results  |  {dataset_type}  |  {completed}/{len(valid_models)} completed  |  {total:.1f}s total")
        print(f"  {'='*80}")
        print(f"  {'Model':<{col_model}} {'Duration':<{col_dur}} URL")
        print(f"  {'-'*col_model} {'-'*col_dur} {'-'*40}")
        for model_name, dur, status, url in rows:
            print(f"  {model_name:<{col_model}} {dur:<{col_dur}} {url}")
        print(f"  {'='*80}\n")

        return results
    
    def run_all_model_comparison(self, dataset_type: str = "comprehensive", direct_agent: bool = False) -> Dict[str, Any]:
        """Test all available models."""
        return self.run_model_comparison_suite(list(self.available_models.keys()), dataset_type, direct_agent)
    
    def list_available_models(self) -> List[str]:
        """List all available models for testing."""
        return list(self.available_models.keys())
    
    def list_available_datasets(self) -> List[str]:
        """List all available dataset types."""
        return ["customer_service", "product_specialist", "promotion_specialist", "comprehensive"]
    

    
    # ===============================================================================
    # DATASET MANAGEMENT METHODS
    # ===============================================================================
    
    def create_all_datasets(self) -> Dict[str, str]:
        """Create all datasets in Datadog. Datadog will auto-version when records change."""
        logger.info("️ Creating all datasets in Datadog...")
        
        results = {}
        dataset_creators = {
            "customer_service": self.create_customer_service_dataset,
            "product_specialist": self.create_product_specialist_dataset,
            "promotion_specialist": self.create_promotion_specialist_dataset,
            "comprehensive": self.create_comprehensive_dataset
        }
        
        for dataset_name, creator_func in dataset_creators.items():
            try:
                dataset = creator_func()
                dataset_id = getattr(dataset, 'id', 'unknown')
                results[dataset_name] = dataset_id
                logger.info(f" Created dataset '{dataset_name}'")
            except Exception as e:
                logger.error(f" Failed to create dataset '{dataset_name}': {e}")
                results[dataset_name] = f"ERROR: {e}"
        
        logger.info(" Dataset creation completed!")
        logger.info(" Datadog will automatically version datasets when records are updated")
        return results
    
    def get_dataset_by_name(self, dataset_type: str) -> Any:
        """Get an existing dataset by name instead of creating a new one."""
        dataset_aliases = {
            "customer_service": "swagbot_customer_service_eval",
            "product_specialist": "swagbot_product_specialist_eval",
            "promotion_specialist": "swagbot_promotion_specialist_eval",
            "comprehensive": "swagbot_comprehensive_eval"
        }
        dataset_name = dataset_aliases.get(dataset_type, dataset_type)
        logger.info(f" Using existing dataset: {dataset_name}")
        return dataset_name
    
    def get_model_info(self, model_key: str = None) -> Dict[str, Any]:
        """Get detailed information about available models."""
        if model_key:
            if model_key not in self.available_models:
                raise ValueError(f"Unknown model: {model_key}")
            return self.available_models[model_key]
        
        return self.available_models
    
    def run_model_comparison(self, model_key: str, dataset_type: str = "comprehensive", direct_agent: bool = False) -> Any:
        """Run a model comparison experiment."""
        test_mode = "direct agent" if direct_agent else "full workflow"
        result, experiment, duration, url = self.run_single_model_experiment(model_key, dataset_type, direct_agent)
        if url:
            print(f"\n  Results: {url}\n")
        return result

def main():
    """Main CLI interface for running model comparison experiments."""
    parser = argparse.ArgumentParser(description="SwagBot LangGraph Model Comparison Experiments")
    
    # Model comparison arguments
    parser.add_argument("--model-comparison", "-m", 
                       help="Run experiment with specific model (e.g., claude-haiku, claude-sonnet)")
    parser.add_argument("--dataset", "-d",
                       default="comprehensive",
                       help="Dataset name or type to use for testing (default: comprehensive). "
                            "Known aliases: customer_service, product_specialist, promotion_specialist, comprehensive. "
                            "Any other value is treated as a raw Datadog dataset name (e.g. product_specialist_benchmark).")
    parser.add_argument("--compare-models", "-c", nargs="+",
                       help="Compare multiple models (e.g., --compare-models claude-haiku claude-sonnet)")
    parser.add_argument("--compare-all-models", "-a", action="store_true",
                       help="Compare all available models")
    parser.add_argument("--direct-agent", action="store_true",
                       help="Test only the specific agent directly (no full workflow, no HTML)")
    
    # Information arguments
    parser.add_argument("--list-models", "-l", action="store_true", 
                       help="List available models")
    parser.add_argument("--list-datasets", action="store_true", 
                       help="List available datasets")
    parser.add_argument("--model-info", 
                       help="Get detailed info about a specific model")
    
    # Dataset management arguments
    parser.add_argument("--create-datasets", action="store_true",
                       help="Create all datasets in Datadog (run this once)")
    parser.add_argument("--create-dataset",
                       choices=["customer_service", "product_specialist", "product_specialist_benchmark",
                                "promotion_specialist", "comprehensive"],
                       help="Create a specific dataset in Datadog")
    
    args = parser.parse_args()
    
    suite = SwagBotModelComparisonSuite()
    
    if args.list_models:
        models = suite.list_available_models()
        print(f" Available {suite.platform.upper()} models for testing:")
        for model_key in models:
            model_config = suite.available_models[model_key]
            print(f" • {model_key}: {model_config['display_name']}")
            print(f"  ID: {model_config['model_id']}")
            print(f"  Description: {model_config['description']}")
            print(f"  Cost Tier: {model_config['cost_tier']}")
            print()
        return
    
    if args.list_datasets:
        datasets = suite.list_available_datasets()
        print(" Available datasets:")
        for dataset in datasets:
            print(f" • {dataset}")
        return
    
    if args.create_datasets:
        print("️ Creating all datasets in Datadog...")
        results = suite.create_all_datasets()
        print("\n Dataset Creation Results:")
        for dataset_name, result in results.items():
            if result.startswith("ERROR"):
                print(f"  {dataset_name}: {result}")
            else:
                print(f"  {dataset_name}: {result}")
        print("\n Datasets created! Datadog will auto-version when records change.")
        return
    
    if args.create_dataset:
        print(f"️ Creating dataset: {args.create_dataset}")
        dataset_creators = {
            "customer_service": suite.create_customer_service_dataset,
            "product_specialist": suite.create_product_specialist_dataset,
            "product_specialist_benchmark": suite.create_product_specialist_benchmark_dataset,
            "promotion_specialist": suite.create_promotion_specialist_dataset,
            "comprehensive": suite.create_comprehensive_dataset
        }
        try:
            dataset = dataset_creators[args.create_dataset]()
            dataset_id = getattr(dataset, 'id', 'unknown')
            print(f" Created dataset '{args.create_dataset}' (id: {dataset_id})")
        except Exception as e:
            print(f" Failed to create dataset '{args.create_dataset}': {e}")
        return
    
    if args.model_info:
        try:
            info = suite.get_model_info(args.model_info)
            print(f" Model Information: {args.model_info}")
            print(f" Display Name: {info['display_name']}")
            print(f" Model ID: {info['model_id']}")
            print(f" Description: {info['description']}")
            print(f" Cost Tier: {info['cost_tier']}")
            print(f" Temperature: {info['temperature']}")
            print(f" Max Tokens: {info['max_tokens']}")
        except ValueError as e:
            print(f" {e}")
        return
    
    if args.compare_all_models:
        test_mode = "direct agent" if args.direct_agent else "full workflow"
        print(f" Comparing all models on {args.dataset} dataset ({test_mode})...")
        suite.run_all_model_comparison(args.dataset, args.direct_agent)
        return
    
    if args.compare_models:
        test_mode = "direct agent" if args.direct_agent else "full workflow"
        pass  # header printed by run_model_comparison_suite
        suite.run_model_comparison_suite(args.compare_models, args.dataset, args.direct_agent)
        return
    
    if args.model_comparison:
        suite.run_model_comparison(args.model_comparison, args.dataset, args.direct_agent)
        return
    
    # No arguments provided - show help
    print(" SwagBot Model Comparison Suite")
    print("=" * 50)
    print("Please provide command-line arguments to run experiments.")
    print("\n Examples:")
    print(" # STEP 1: Create datasets in Datadog (run once):")
    print(" python swagbot_experiments.py --create-datasets")
    print()
    print(" # STEP 2: Run experiments:")
    print(" python swagbot_experiments.py --compare-all-models --dataset product_specialist --direct-agent")
    print()
    print(" # Dataset management:")
    print(" python swagbot_experiments.py --create-dataset product_specialist  # Create/update single dataset")
    print(" python swagbot_experiments.py --list-datasets                        # List available datasets")
    print()
    print(" # Compare specific models:")
    print(" python swagbot_experiments.py --compare-models claude-haiku claude-sonnet --dataset product_specialist --direct-agent")
    print()
    print(" # Test single model:")
    print(" python swagbot_experiments.py --model-comparison claude-haiku --dataset product_specialist --direct-agent")
    print()
    print(" # Model information:")
    print(" python swagbot_experiments.py --list-models")
    print()
    print(" Note: Datadog automatically versions datasets when records are updated")
    print(" For full help: python swagbot_experiments.py --help")
    return

if __name__ == "__main__":
    main()
