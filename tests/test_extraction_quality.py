from pipeline.extraction_quality import coverage_issues, deterministic_hints, merge_missing

SMARTOPTICS = """
SO-TQSFP-DD-4CC-ZR
QSFP-DD OIF 400G-ZR Ethernet Coh Tunable Flexgrid 120km LC D9128-D9612
The transmitter is tunable over the ITU C-Band at 100 GHz grid.
Transmission media SM (2x LC)
Nominal wavelengths 191.28 - 196.12THz
Power consumption < 17 W (Class 8)
Operating temperature 0°C to +70°C
400Gbps Ethernet applications. 60Gbaud and 16QAM coherent modulation.
CMIS4.0, OIF400ZR, QSFP-DD MSA
"""

NOKIA = """
Nokia 400ZR/ZR+ Coherent Modules
QDDCO4Z/QDDCO4/QDDCO4H provide coherent optics in QSFP-DD modules.
Supporting 100G to 400G capacities. The QDDCO4Z is based on OIF 400ZR and 16QAM.
The QDDCO4 supports OpenZR+ and supports QPSK, 8QAM and 16QAM.
"""


def test_smartoptics_explicit_recovery():
    h = deterministic_hints(SMARTOPTICS)
    assert "QSFP-DD" in h["form_factor"]
    assert h["frequency_min_thz"] == [191.28]
    assert h["frequency_max_thz"] == [196.12]
    assert 100.0 in h["channel_spacing_ghz"]
    assert h["temp_min_c"] == 0.0
    assert h["temp_max_c"] == 70.0
    assert 400.0 in h["bit_rate_gbps"]
    assert "16QAM" in h["modulation_format"]


def test_nokia_is_not_vendor_hardcoded():
    h = deterministic_hints(NOKIA)
    assert "QSFP-DD" in h["form_factor"]
    assert {"QPSK", "8QAM", "16QAM"}.issubset(set(h["modulation_format"]))
    assert 100.0 in h["bit_rate_gbps"]
    assert 400.0 in h["bit_rate_gbps"]


def test_coverage_rejects_silent_null_success():
    issues = coverage_issues(SMARTOPTICS, {})
    assert any("form_factor" in x for x in issues)
    assert any("frequency_min_thz" in x for x in issues)
    assert any("channel_spacing_ghz" in x for x in issues)


def test_fallback_does_not_overwrite_llm_value():
    merged = merge_missing({"form_factor": ["OSFP"], "frequency_min_thz": None}, {"form_factor": ["QSFP-DD"], "frequency_min_thz": [191.28]})
    assert merged["form_factor"] == ["OSFP"]
    assert merged["frequency_min_thz"] == [191.28]
