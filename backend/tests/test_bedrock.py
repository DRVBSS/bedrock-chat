import os
import sys

import boto3
from ulid import ULID

os.environ["REGION"] = "us-west-2"
os.environ["BEDROCK_REGION"] = "us-west-2"
os.environ["ENABLE_BEDROCK_GLOBAL_INFERENCE"] = "true"
os.environ["ENABLE_BEDROCK_CROSS_REGION_INFERENCE"] = "true"

sys.path.append(".")

import unittest
from pprint import pprint
from unittest.mock import patch

from app.bedrock import (
    call_converse_api,
    compose_args_for_converse_api,
    generation_params_to_converse_configuration,
    get_model_id,
    is_temperature_supported,
    is_top_k_supported,
)
from app.repositories.models.custom_bot import GenerationParamsModel
from app.repositories.models.conversation import SimpleMessageModel, TextContentModel
from app.repositories.models.custom_bot_guardrails import BedrockGuardrailsModel
from app.routes.schemas.conversation import type_model_name

# MODEL: type_model_name = "claude-v3-haiku"
MODEL: type_model_name = "claude-v3.7-sonnet"


class TestGetModelId(unittest.TestCase):
    def test_get_model_id_with_cross_region_supported_model(self):
        model = "claude-v3.5-sonnet"
        # Prefix with "us." to enable cross-region
        expected_model_id = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
        self.assertEqual(
            get_model_id(
                model,
                enable_global=True,
                enable_cross_region=True,
                bedrock_region="us-east-1",
            ),
            expected_model_id,
        )

    def test_get_model_id_without_cross_region(self):
        model = "claude-v3.5-sonnet"
        # No prefix to disable cross-region
        expected_model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        self.assertEqual(
            get_model_id(
                model,
                enable_global=False,
                enable_cross_region=False,
                bedrock_region="us-east-1",
            ),
            expected_model_id,
        )

    def test_get_model_id_with_unsupported_region_for_cross_region(self):
        model = "claude-v3.5-sonnet"
        # Cross region is disabled because the region is not supported
        expected_model_id = "apac.anthropic.claude-3-5-sonnet-20240620-v1:0"
        self.assertEqual(
            get_model_id(
                model,
                enable_global=True,
                enable_cross_region=True,
                bedrock_region="ap-northeast-1",
            ),
            expected_model_id,
        )

    def test_get_model_id_with_global_inference_priority(self):
        """Global inference is selected for supported model and region"""
        model = "claude-v4-sonnet"
        expected_model_id = "global.anthropic.claude-sonnet-4-20250514-v1:0"
        self.assertEqual(
            get_model_id(
                model,
                enable_global=True,
                enable_cross_region=True,
                bedrock_region="us-east-1",
            ),
            expected_model_id,
        )

    def test_get_model_id_without_global_with_cross_region_inference_priority(self):
        """Global inference is selected for supported model and region"""
        model = "claude-v4-sonnet"
        expected_model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
        self.assertEqual(
            get_model_id(
                model,
                enable_global=False,
                enable_cross_region=True,
                bedrock_region="us-east-1",
            ),
            expected_model_id,
        )

    def test_get_model_id_with_regional_fallback(self):
        """Falls back to regional cross-region for non-global supported models"""
        model = "claude-v3.5-sonnet"  # Non-global supported
        expected_model_id = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
        self.assertEqual(
            get_model_id(
                model,
                enable_global=True,
                enable_cross_region=True,
                bedrock_region="us-east-1",
            ),
            expected_model_id,
        )

    def test_get_model_id_with_unsupported_region_fallback(self):
        """Falls back to regional for global-supported model in unsupported region"""
        model = "claude-v4-sonnet"
        # ap-south-1 is not global-supported but APAC regional-supported
        expected_model_id = "apac.anthropic.claude-sonnet-4-20250514-v1:0"
        self.assertEqual(
            get_model_id(
                model,
                enable_global=True,
                enable_cross_region=True,
                bedrock_region="ap-south-1",
            ),
            expected_model_id,
        )


class TestCallConverseApi(unittest.TestCase):
    def test_call_converse_api_with_global_inference(self):
        """Actual LLM call using global inference profile"""
        message = SimpleMessageModel(
            role="user",
            content=[
                TextContentModel(
                    content_type="text",
                    body="Hello! Please respond with just 'Global inference works!'",
                )
            ],
        )

        # Use global inference supported model
        arg = compose_args_for_converse_api(
            [message],
            "claude-v4-sonnet",  # Global inference supported
            stream=False,
        )

        # Verify modelId is global profile
        expected_model_id = "global.anthropic.claude-sonnet-4-20250514-v1:0"
        self.assertEqual(arg["modelId"], expected_model_id)

        # Actual API call
        response = call_converse_api(arg)

        # Verify basic response structure
        self.assertIn("output", response)
        self.assertIn("message", response["output"])
        self.assertIn("content", response["output"]["message"])

        print(f"Global inference response: {response['output']['message']['content']}")

    def test_call_converse_api_with_regional_fallback(self):
        """Actual LLM call with regional cross-region fallback"""
        message = SimpleMessageModel(
            role="user",
            content=[
                TextContentModel(
                    content_type="text",
                    body="Hello! Please respond with just 'Regional inference works!'",
                )
            ],
        )

        # Use non-global supported model
        arg = compose_args_for_converse_api(
            [message],
            "claude-v3.5-sonnet",  # Non-global supported, regional supported
            stream=False,
        )

        # Verify modelId is regional profile
        expected_model_id = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
        self.assertEqual(arg["modelId"], expected_model_id)

        # Actual API call
        response = call_converse_api(arg)

        # Verify basic response structure
        self.assertIn("output", response)
        self.assertIn("message", response["output"])
        self.assertIn("content", response["output"]["message"])

        print(
            f"Regional inference response: {response['output']['message']['content']}"
        )

    def test_call_converse_api(self):
        message = SimpleMessageModel(
            role="user",
            content=[
                TextContentModel(
                    content_type="text",
                    body="Hello, World!",
                )
            ],
        )
        arg = compose_args_for_converse_api(
            [message],
            MODEL,
            stream=False,
        )

        response = call_converse_api(arg)
        pprint(response)


class TestCallConverseApiWithGuardrails(unittest.TestCase):
    def setUp(self):
        # Note that the region must be the same as the one used in the bedrock client
        # https://github.com/aws/aws-sdk-js-v3/issues/6482
        self.bedrock_client = boto3.client("bedrock", region_name="us-east-1")
        self.guardrail_name = f"test-guardrail-{ULID()}"

        # Create dummy guardrail
        res = self.bedrock_client.create_guardrail(
            name=self.guardrail_name,
            description="Test guardrail for unit tests",
            contentPolicyConfig={
                "filtersConfig": [
                    {"type": "SEXUAL", "inputStrength": "LOW", "outputStrength": "LOW"},
                ]
            },
            blockedInputMessaging="blocked",
            blockedOutputsMessaging="blocked",
        )

        res_ver = self.bedrock_client.create_guardrail_version(
            guardrailIdentifier=res["guardrailArn"],
        )

        self.guardrail = BedrockGuardrailsModel(
            is_guardrail_enabled=True,
            hate_threshold=0,
            insults_threshold=0,
            sexual_threshold=1,
            violence_threshold=0,
            misconduct_threshold=0,
            grounding_threshold=0,
            relevance_threshold=0,
            guardrail_arn=res["guardrailArn"],
            guardrail_version=res_ver["version"],
            # guardrail_version="DRAFT",
        )
        self.guardrail_arn = res["guardrailArn"]

    def tearDown(self):
        print("Cleaning up...")
        # Delete dummy guardrail
        try:
            self.bedrock_client.delete_guardrail(guardrailIdentifier=self.guardrail_arn)

        except Exception as e:
            print(f"Error deleting guardrail: {e}")

    def test_call_converse_api_with_guardrails(self):
        message = SimpleMessageModel(
            role="user",
            content=[
                TextContentModel(
                    content_type="text",
                    body="Hello, World!",
                )
            ],
        )
        arg = compose_args_for_converse_api(
            [message],
            MODEL,
            guardrail=self.guardrail,
            stream=False,
        )

        pprint(arg)

        response = call_converse_api(arg)
        pprint(response)


class TestIsTopKSupported(unittest.TestCase):
    """Claude Opus 4.7 deprecated top_k. Older Claude models still accept it.

    Regression guard for: ValidationException ("`top_k` is deprecated for
    this model") thrown by Bedrock when Opus 4.7 receives top_k in
    additionalModelRequestFields.
    """

    def test_opus_4_7_does_not_support_top_k(self):
        self.assertFalse(is_top_k_supported("claude-v4.7-opus"))

    def test_opus_4_6_still_supports_top_k(self):
        self.assertTrue(is_top_k_supported("claude-v4.6-opus"))

    def test_sonnet_4_6_still_supports_top_k(self):
        self.assertTrue(is_top_k_supported("claude-v4.6-sonnet"))

    def test_haiku_4_5_still_supports_top_k(self):
        self.assertTrue(is_top_k_supported("claude-v4.5-haiku"))


class TestGenerationParamsConverseConfigTopK(unittest.TestCase):
    """End-to-end check that top_k is stripped from the payload for Opus 4.7
    in the non-reasoning code path (the path Bedrock Chat hits by default
    when a user picks a model from the dropdown and asks a question).
    """

    def _params(self):
        return GenerationParamsModel(
            max_tokens=2000,
            top_k=250,
            top_p=0.999,
            temperature=0.6,
            stop_sequences=[],
            reasoning_params=None,
        )

    def test_opus_4_7_omits_top_k_in_additional_fields(self):
        cfg = generation_params_to_converse_configuration(
            model="claude-v4.7-opus",
            generation_params=self._params(),
            enable_reasoning=False,
        )
        self.assertNotIn(
            "top_k", cfg.get("additionalModelRequestFields", {})
        )

    def test_opus_4_6_keeps_top_k_in_additional_fields(self):
        cfg = generation_params_to_converse_configuration(
            model="claude-v4.6-opus",
            generation_params=self._params(),
            enable_reasoning=False,
        )
        self.assertEqual(
            cfg["additionalModelRequestFields"]["top_k"], 250
        )


class TestIsTemperatureSupported(unittest.TestCase):
    """Claude Opus 4.7 also deprecated user-controlled temperature.

    Regression guard for: ValidationException ("`temperature` is deprecated
    for this model") thrown by Bedrock when Opus 4.7 receives temperature
    in inferenceConfig.
    """

    def test_opus_4_7_does_not_support_temperature(self):
        self.assertFalse(is_temperature_supported("claude-v4.7-opus"))

    def test_opus_4_6_still_supports_temperature(self):
        self.assertTrue(is_temperature_supported("claude-v4.6-opus"))

    def test_sonnet_4_6_still_supports_temperature(self):
        self.assertTrue(is_temperature_supported("claude-v4.6-sonnet"))


class TestGenerationParamsConverseConfigTemperature(unittest.TestCase):
    """temperature must be absent from inferenceConfig for Opus 4.7 across
    all three upstream code paths (non-reasoning, adaptive-thinking,
    explicit-reasoning). The post-build cleanup is the single source of
    truth for this guarantee.
    """

    def _params(self):
        return GenerationParamsModel(
            max_tokens=2000,
            top_k=250,
            top_p=0.999,
            temperature=0.6,
            stop_sequences=[],
            reasoning_params=None,
        )

    def test_opus_4_7_omits_temperature_no_reasoning(self):
        cfg = generation_params_to_converse_configuration(
            model="claude-v4.7-opus",
            generation_params=self._params(),
            enable_reasoning=False,
        )
        self.assertNotIn("temperature", cfg["inferenceConfig"])

    def test_opus_4_7_omits_temperature_with_adaptive_thinking(self):
        cfg = generation_params_to_converse_configuration(
            model="claude-v4.7-opus",
            generation_params=self._params(),
            enable_reasoning=True,
        )
        self.assertNotIn("temperature", cfg["inferenceConfig"])

    def test_opus_4_6_keeps_temperature_no_reasoning(self):
        cfg = generation_params_to_converse_configuration(
            model="claude-v4.6-opus",
            generation_params=self._params(),
            enable_reasoning=False,
        )
        self.assertIn("temperature", cfg["inferenceConfig"])


if __name__ == "__main__":
    unittest.main()
