"""Synthetic pipeline integration tests, never used as optical training data."""
import tempfile
import unittest
from pathlib import Path
from otse_auto.core import GROUPS, LABEL_PROVENANCE, write_json, read_jsonl
from otse_auto.dataset import build, token_record
from test_safety import CharacterTokenizer


class BuildTests(unittest.TestCase):
    def test_machine_pipeline_can_build_without_approved_flags(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            for split, n in [('train', 50), ('dev', 12)]:
                for i in range(n):
                    for group, names in GROUPS.items():
                        docid = f'{split}-{i:03d}'
                        fields = {f: {'state': 'unresolved', 'value': None, 'reason': 'Unknown synthetic fixture.'} for f in names}
                        field, value, quote = {'identity_and_physical': ('vendor', 'TESTCO', 'TESTCO'),
                            'interfaces_and_spectrum': ('wavelength_center_nm', [850], 'Wavelength 850 nm'),
                            'optical_performance_and_fec': ('bit_rate_gbps', [1.25], 'Bit rate 1.25 Gbps')}[group]
                        fields[field] = {'state': 'supported', 'value': value, 'reason': 'Synthetic test only.',
                                         'evidence': [{'block_id': 'P001B00000', 'page': 1, 'quote': quote}]}
                        ann = {'document_id': docid, 'sha256': docid, 'split': split, 'family_id': docid,
                               'field_group': group, 'label_provenance': LABEL_PROVENANCE, 'fields': fields,
                               'blocks': [{'id': 'P001B00000', 'page': 1, 'text': quote, 'start_line': 0}]}
                        write_json(out / 'annotations' / split / f'{docid}__{group}.json', ann)
            report = build(out, CharacterTokenizer(), 30000)
            self.assertTrue(report['ready_for_training'])
            self.assertEqual(report['splits']['train']['documents'], 50)
            self.assertEqual(report['splits']['dev']['documents'], 12)
            self.assertEqual(report['splits']['train']['examples'], 150)
            self.assertEqual(report['splits']['dev']['examples'], 36)
            self.assertFalse(report['human_verified'])
            for record in read_jsonl(out / 'dataset/train.jsonl'):
                self.assertNotIn('review_status', record)
                token_record(record, CharacterTokenizer(), 30000)

    def test_empty_machine_dataset_does_not_train(self):
        with tempfile.TemporaryDirectory() as d:
            report = build(Path(d), CharacterTokenizer(), 30000)
            self.assertFalse(report['ready_for_training'])
            self.assertEqual(report['splits']['train']['examples'], 0)


if __name__ == '__main__':
    unittest.main()
