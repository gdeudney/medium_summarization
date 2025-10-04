import pytest
from prompt_opt.engine import PromptOptimizationEngine
from prompt_opt.config import EngineConfig, DatasetConfig

def test_extract_new_prompt_valid():
    """Tests that a valid prompt is extracted correctly."""
    raw_text = "Here is some text. <prompt>This is the new prompt.</prompt> And some more text."
    prompt = PromptOptimizationEngine._extract_new_prompt(raw_text)
    assert prompt == "This is the new prompt."

def test_extract_new_prompt_no_tag():
    """Tests that None is returned when no prompt tag is present."""
    raw_text = "The optimizer failed to provide a tag."
    prompt = PromptOptimizationEngine._extract_new_prompt(raw_text)
    assert prompt is None

def test_extract_new_prompt_with_newlines():
    """Tests that newlines inside the tag are preserved."""
    raw_text = "<prompt>\nLine 1\nLine 2\n</prompt>"
    prompt = PromptOptimizationEngine._extract_new_prompt(raw_text)
    assert prompt == "Line 1\nLine 2"

def test_config_validation_fails_on_missing_path(tmp_path):
    """Tests that Pydantic validation catches a non-existent dataset path."""
    with pytest.raises(ValueError, match="Dataset path does not exist"):
        DatasetConfig(path=tmp_path / "non_existent_folder")