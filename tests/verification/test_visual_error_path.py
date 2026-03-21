
import pytest
from unittest.mock import MagicMock, AsyncMock
from swarm.verification.visual import VisualVerifier, VisualIssue
from swarm.config.models import VerificationConfig

@pytest.mark.asyncio
async def test_visual_issue_validation_error_path():
    """
    Test that VisualVerifier handles invalid issue schemas gracefully
    by skipping them and continuing with valid ones.
    """
    # Setup mock config
    config = MagicMock(spec=VerificationConfig)
    config.pass_threshold = 0.8
    config.vision_model = "gpt-4o"

    verifier = VisualVerifier(config)

    # Mock _capture_screenshot to return bytes
    verifier._capture_screenshot = AsyncMock(return_value=b"fake_screenshot")

    # Mock _run_vision_critique to return a dict with some invalid issues
    # and some valid issues
    mock_critique = {
        "score": 0.9,
        "summary": "Test summary",
        "issues": [
            {
                "severity": "major",
                "category": "alignment",
                "element": ".header",
                "description": "Valid issue",
                "suggestion": "Fix it"
            },
            {
                "invalid_field": "This should fail validation"
            },
            {
                "severity": "minor",
                "category": "spacing",
                "element": ".footer",
                "description": "Another valid issue",
                "suggestion": "Fix it too"
            }
        ]
    }

    verifier._run_vision_critique = AsyncMock(return_value=mock_critique)

    # We also need to mock setup since it tries to import playwright
    verifier.setup = AsyncMock(return_value=None)

    # Run verify
    result = await verifier.verify(url="http://example.com", requirements="Test requirements")

    # Assertions
    assert result.passed is True
    # Should have 2 valid issues, the invalid one should be skipped
    assert len(result.issues) == 2
    assert result.issues[0].description == "Valid issue"
    assert result.issues[1].description == "Another valid issue"
    assert isinstance(result.issues[0], VisualIssue)
    assert isinstance(result.issues[1], VisualIssue)
