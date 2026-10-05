import pytest

from app.services.gemini_service import GeminiLLMService


# ============================================================
# TEST HELPERS
# ============================================================

class FakeResponse:
    """Fake Gemini API response."""

    def __init__(self, text):
        self.text = text


class FakeModels:
    """Fake Gemini models client."""

    def __init__(self, response_text):
        self.response_text = response_text
        self.last_model = None
        self.last_contents = None
        self.last_config = None

    def generate_content(
        self,
        model,
        contents,
        config=None,
    ):
        self.last_model = model
        self.last_contents = contents
        self.last_config = config

        return FakeResponse(
            self.response_text
        )


class FakeClient:
    """Fake Gemini client."""

    def __init__(self, response_text):
        self.models = FakeModels(
            response_text
        )


# ============================================================
# GEMINI SERVICE TESTS
# ============================================================

def test_gemini_service_requires_api_key(
    monkeypatch,
):
    """Service should reject a missing API key."""

    monkeypatch.delenv(
        "GEMINI_API_KEY",
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="GEMINI_API_KEY is not configured.",
    ):
        GeminiLLMService()


def test_gemini_service_can_be_created(
    monkeypatch,
):
    """Service should initialize with an API key."""

    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-api-key",
    )

    service = GeminiLLMService()

    assert service is not None
    assert service.client is not None


def test_gemini_generate_with_mocked_client(
    monkeypatch,
):
    """Generate an answer without calling Gemini."""

    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-api-key",
    )

    service = GeminiLLMService()

    expected_answer = (
        "Research methodology explains the "
        "systematic methods used to conduct "
        "a research study."
    )

    fake_client = FakeClient(
        expected_answer
    )

    service.client = fake_client

    answer = service.generate(
        question=(
            "What is the main purpose "
            "of a research methodology?"
        ),
        context=(
            "Research methodology describes "
            "the systematic approach used to "
            "collect, analyse, and interpret data."
        ),
    )

    assert answer == expected_answer

    assert (
        fake_client.models.last_model
        == "gemini-3.6-flash"
    )

    assert (
        "What is the main purpose"
        in fake_client.models.last_contents
    )

    assert (
        "Research methodology describes"
        in fake_client.models.last_contents
    )

    assert (
        fake_client.models.last_config
        is not None
    )


def test_gemini_generate_returns_expected_text(
    monkeypatch,
):
    """Verify that Gemini response text is returned."""

    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-api-key",
    )

    service = GeminiLLMService()

    expected_answer = (
        "The methodology provides a systematic "
        "framework for conducting research."
    )

    service.client = FakeClient(
        expected_answer
    )

    result = service.generate(
        question="What is research methodology?",
        context=(
            "Research methods and procedures."
        ),
    )

    assert result == expected_answer


def test_gemini_generate_rejects_empty_question(
    monkeypatch,
):
    """Empty questions should raise ValueError."""

    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-api-key",
    )

    service = GeminiLLMService()

    with pytest.raises(
        ValueError,
        match="Question cannot be empty.",
    ):
        service.generate(
            question="",
            context="Some research context.",
        )


def test_gemini_generate_handles_empty_context(
    monkeypatch,
):
    """Empty context should not call the Gemini API."""

    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-api-key",
    )

    service = GeminiLLMService()

    fake_client = FakeClient(
        "This should not be returned."
    )

    service.client = fake_client

    question = (
        "What is the main purpose "
        "of research methodology?"
    )

    result = service.generate(
        question=question,
        context="",
    )

    assert result == (
        "I cannot find any source information "
        "to answer: "
        "'What is the main purpose of research "
        "methodology?'"
    )

    assert (
        fake_client.models.last_model
        is None
    )