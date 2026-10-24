"""The PHI redaction gate, carried over in the shared skeleton."""

from app import safety


def test_redacts_email_and_phone():
    text = "Patient: Jane Doe, email jane@example.com, phone 661-555-0142"
    redacted, report = safety.redact_phi(text)
    assert "jane@example.com" not in redacted
    assert "661-555-0142" not in redacted
    assert report.get("EMAIL") == 1


def test_redacts_ssn_and_dob():
    text = "SSN 123-45-6789, DOB: 03/04/1950"
    redacted, report = safety.redact_phi(text)
    assert "123-45-6789" not in redacted
    assert "03/04/1950" not in redacted


def test_clean_text_passes_through():
    text = "left heel wound, week three photo"
    redacted, report = safety.redact_phi(text)
    assert redacted == text
    assert report == {}
