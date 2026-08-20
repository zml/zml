import export_tokenizer_reference as reference


def test_tokenizer_corpus_covers_text_classes_and_literal_controls() -> None:
    names = {name for name, _ in reference.ORDINARY_TEXT_CASES}
    assert names == {"empty", "ascii", "unicode", "whitespace", "numbers", "han", "literal_controls"}
    literal = dict(reference.ORDINARY_TEXT_CASES)["literal_controls"]
    assert "<|open|>" in literal and "<|end_of_msg|>" in literal


def test_chat_corpus_covers_literal_control_injection() -> None:
    prompts = dict(reference.CHAT_PROMPTS)
    assert set(prompts) == {"ascii", "unicode", "literal_controls"}
    assert "<|open|>" in prompts["literal_controls"]


def test_fuzz_corpus_is_deterministic_and_diverse() -> None:
    first = reference.fuzz_text_cases()
    assert first == reference.fuzz_text_cases()
    assert len(first) == 512
    assert len({text for _, text in first}) == 512
    assert any("世界" in text for _, text in first)
    assert any("\r\n" in text for _, text in first)
