from pipeline.atom_extractor import FIELD_GROUPS


def test_atomic_groups_do_not_request_untrained_metadata():
    all_fields = {field for _, fields in FIELD_GROUPS for field in fields}
    assert "notes" not in all_fields
    assert "provenance_datasheet_sections" not in all_fields


def test_expected_group_sizes():
    sizes = {name: len(fields) for name, fields in FIELD_GROUPS}
    assert sizes == {
        "identity_and_physical": 9,
        "interfaces_and_spectrum": 16,
        "optical_performance_and_fec": 23,
    }
