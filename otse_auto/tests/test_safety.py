"""Offline safety/regression tests. No model or endpoint calls, no training fixtures."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from otse_auto.core import GROUPS, validate_value, inside, digest_obj, messages_for
from otse_auto.evidence import choose_blocks, serialize_blocks, alternate_order, quote_present, fit_blocks
from otse_auto.labeling import validate_proposal, consensus, semantic_anchor_gate, numeric_values, value_has_lexical_support, Ollama
from otse_auto.dataset import render_answer, token_record, make_record, accepted_fields
from otse_auto.evaluation import parse_object, bootstrap_delta
from otse_auto import scoring


class CharacterTokenizer:
    """Test-double with exact offsets; NOT a stand-in for measured Qwen token counts."""
    is_fast = True
    pad_token_id = 0
    eos_token_id = 3
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = ''.join('<' + m['role'] + '>' + m['content'] + '</end>' for m in messages)
        if add_generation_prompt:
            text += '<assistant>'
        return [ord(c) for c in text] if tokenize else text
    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(c) for c in text], 'offset_mapping': [(i, i + 1) for i in range(len(text))]}


def block(text='Receiver sensitivity  -18  dBm', ident='P003B00000', page=3):
    return {'id': ident, 'page': page, 'start_line': 0, 'text': text}


def supported(value, quote='Receiver sensitivity  -18  dBm', ident='P003B00000'):
    return {'state': 'supported', 'value': value, 'evidence': [{'block_id': ident, 'quote': quote}], 'reason': 'Explicit labeled row and units.'}


def fixture():
    group = 'identity_and_physical'
    fields = {f: {'state': 'unresolved', 'value': None, 'reason': 'Unknown test value'} for f in GROUPS[group]}
    fields['vendor'] = {'state': 'supported', 'value': 'TESTCO', 'evidence': [{'block_id': 'P001B00000', 'page': 1, 'quote': 'TESTCO'}], 'reason': 'Test fixture only.'}
    fields['temp_min_c'] = {'state': 'absent_in_input', 'value': None, 'reason': 'Test fixture only.'}
    return {'document_id': 'fixture', 'sha256': 'test-sha', 'family_id': 'test-family', 'split': 'train',
            'field_group': group, 'fields': fields, 'blocks': [block('TESTCO product information', 'P001B00000', 1)]}


class SafetyTests(unittest.TestCase):
    def test_schema_has_48_unique_fields(self):
        fields = [f for group in GROUPS.values() for f in group]
        self.assertEqual(len(fields), 48)
        self.assertEqual(len(set(fields)), 48)

    def test_exact_quote_accepts_whitespace_only_change(self):
        self.assertTrue(quote_present('Receiver sensitivity -18 dBm', block()['text']))

    def test_quote_sign_not_normalized_away(self):
        self.assertFalse(quote_present('Receiver sensitivity 18 dBm', block()['text']))

    def test_true_supported_negative_number(self):
        value, reason = validate_proposal('rx_sensitivity_dbm', supported([-18.0]), [block()])
        self.assertEqual(value['value'], [-18.0])
        self.assertFalse(reason)

    def test_wrong_number_rejected_even_with_real_quote(self):
        value, reason = validate_proposal('rx_sensitivity_dbm', supported([-17.0]), [block()])
        self.assertIsNone(value)
        self.assertEqual(reason, 'value_not_lexically_supported')

    def test_wrong_block_rejected(self):
        proposal = supported([-18.0], ident='P009B00000')
        self.assertIsNone(validate_proposal('rx_sensitivity_dbm', proposal, [block()])[0])

    def test_invented_quote_rejected(self):
        self.assertIsNone(validate_proposal('rx_sensitivity_dbm', supported([-18], quote='Imaginary sensitivity -18 dBm'), [block()])[0])

    def test_unreviewed_is_not_accepted(self):
        value, _ = validate_proposal('vendor', {'value': 'TESTCO', 'status': 'UNREVIEWED'}, [block()])
        self.assertIsNone(value)

    def test_no_null_from_ambiguous(self):
        a = {'fields': {'rx_sensitivity_dbm': {'state': 'ambiguous', 'value': None, 'reason': 'Conflicting variants'}}}
        result = consensus('optical_performance_and_fec', a, a, [block()])
        self.assertEqual(result['rx_sensitivity_dbm']['state'], 'unresolved')

    def test_two_pass_disagreement_masked(self):
        a = {'fields': {'rx_sensitivity_dbm': supported([-18])}}
        b = {'fields': {'rx_sensitivity_dbm': {'state': 'absent_in_input', 'value': None, 'reason': 'Not seen'}}}
        result = consensus('optical_performance_and_fec', a, b, [block()])
        self.assertEqual(result['rx_sensitivity_dbm']['state'], 'unresolved')

    def test_missing_field_not_assumed_absent(self):
        result = consensus('identity_and_physical', {'fields': {}}, {'fields': {}}, [])
        self.assertTrue(all(r['state'] == 'unresolved' for r in result.values()))

    def test_nonempty_absence_rejected(self):
        item = {'state': 'absent_in_input', 'value': ['LC'], 'reason': 'No optical connector'}
        self.assertIsNone(validate_proposal('connector_type', item, [block()])[0])

    def test_bools_not_numeric(self):
        with self.assertRaises(ValueError):
            validate_value('temp_min_c', True)

    def test_scalar_list_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            validate_value('rx_sensitivity_dbm', -18)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            validate_value('rx_sensitivity_dbm', [float('inf')])

    def test_units_adjacent_to_numbers(self):
        self.assertEqual(numeric_values('1.25Gbps -18dBm 5e-5'), [1.25, -18.0, 5e-5])

    def test_deterministic_reach_conversion(self):
        self.assertTrue(value_has_lexical_support('reach_km', [0.55], ['Maximum reach 550 m']))

    def test_no_wavelength_frequency_conversion(self):
        self.assertFalse(value_has_lexical_support('frequency_min_thz', [193.1], ['Wavelength 1552.52 nm']))

    def test_no_baud_from_bitrate(self):
        self.assertFalse(semantic_anchor_gate('baud_rate_gbaud', [25], ['Data rate 25 Gbps']))

    def test_connector_not_media_interface(self):
        self.assertFalse(semantic_anchor_gate('media_interface_name', ['LC'], ['Duplex LC connector']))

    def test_form_factor_not_host_interface(self):
        self.assertFalse(semantic_anchor_gate('host_interface_name', ['QSFP-DD'], ['QSFP-DD module']))

    def test_id_requires_semantic_anchor(self):
        self.assertFalse(semantic_anchor_gate('host_interface_id', [17], ['Address 17 memory contents']))
        self.assertTrue(semantic_anchor_gate('host_interface_id', [17], ['Host electrical interface ID 17']))

    def test_storage_temperature_not_operating(self):
        self.assertFalse(semantic_anchor_gate('temp_min_c', -40, ['Storage temperature -40 to 85 C']))

    def test_unknown_spans_are_all_masked(self):
        ann = fixture()
        record = make_record(ann, CharacterTokenizer(), 30000)
        encoded = token_record(record, CharacterTokenizer(), 30000)
        prefix = CharacterTokenizer().apply_chat_template(record['messages'][:-1], add_generation_prompt=True)
        for start, end in record['unknown_answer_spans']:
            self.assertTrue(all(v == -100 for v in encoded['labels'][len(prefix) + start:len(prefix) + end]))
        self.assertTrue(any(v != -100 for v in encoded['labels']))
        self.assertTrue(all(v == -100 for v in encoded['labels'][:len(prefix)]))

    def test_unknown_fields_follow_known_fields(self):
        ann = fixture()
        text, spans, unknown = render_answer(ann['field_group'], ann['fields'], ['vendor', 'temp_min_c'])
        self.assertLess(max(b for a, b in spans), min(a for a, b in unknown))
        self.assertEqual(set(json.loads(text)), set(GROUPS[ann['field_group']]))

    def test_overlength_is_rejected_without_truncation(self):
        with self.assertRaises(ValueError):
            make_record(fixture(), CharacterTokenizer(), 50)

    def test_no_supervision_when_no_known_fields(self):
        ann = fixture()
        for row in ann['fields'].values():
            row['state'] = 'unresolved'
        self.assertIsNone(make_record(ann, CharacterTokenizer(), 30000))

    def test_negatives_are_capped_for_training(self):
        ann = fixture()
        for name in ann['fields']:
            if name != 'vendor':
                ann['fields'][name] = {'state': 'absent_in_input', 'value': None, 'reason': 'Fixture'}
        self.assertEqual(len(accepted_fields(ann, balance=True)), 4)

    def test_augmentation_preserves_whole_blocks(self):
        blocks = [block('alpha table', 'A', 1), block('beta table', 'B', 2), block('gamma table', 'C', 3)]
        alt = alternate_order(blocks)
        self.assertNotEqual(alt, blocks)
        self.assertEqual({b['id']: b['text'] for b in blocks}, {b['id']: b['text'] for b in alt})

    def test_retrieval_keeps_page_numbers(self):
        pages = ['Manufacturer\nModel\n' + 'General identity\n' * 12,
                 'Receiver sensitivity -18 dBm\nMaximum receiver overload +3 dBm\n' * 12]
        blocks = choose_blocks(pages, 'optical_performance_and_fec', 7000)
        self.assertIn(2, [b['page'] for b in blocks])
        self.assertIn('### PAGE 2 BLOCK', serialize_blocks(blocks))

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                inside(Path(d), '../escape')

    def test_malformed_output_not_empty_correct_answer(self):
        self.assertIsNone(parse_object('Here is your answer: {}'))
        self.assertIsNone(parse_object('{"temp_min_c":NaN}'))
        counts = scoring.score_one({'field': 'vendor', 'gold_value': None}, scoring.MISSING)['counts']
        self.assertEqual(counts['correct'], 0)
        self.assertEqual(counts['missing'], 1)

    def test_f1_reuse_correct(self):
        a = scoring.score_one({'field': 'rx_sensitivity_dbm', 'gold_value': [-18]}, [-18, -17])['counts']
        self.assertEqual((a['tp'], a['fp'], a['fn']), (1, 1, 0))
        self.assertAlmostEqual(scoring.metrics(a)['micro_f1'], 2 / 3)

    def test_bootstrap_reproducible(self):
        a = scoring.score_one({'field': 'vendor', 'gold_value': 'A'}, 'A')['counts']
        b = scoring.score_one({'field': 'vendor', 'gold_value': 'A'}, 'B')['counts']
        x = bootstrap_delta({'x': b, 'y': a}, {'x': a, 'y': a}, reps=50)
        self.assertEqual(x, bootstrap_delta({'x': b, 'y': a}, {'x': a, 'y': a}, reps=50))

    def test_ollama_success_is_cached(self):
        with tempfile.TemporaryDirectory() as d:
            client = Ollama('http://local', Path(d), CharacterTokenizer(), 20000)
            client.models = {'testmodel': {'digest': 'abc'}}
            response = {'done': True, 'done_reason': 'stop', 'message': {'content': '{"fields":{}}'}}
            with patch('otse_auto.labeling.http_json', return_value=response) as request:
                first = client.ask('testmodel', [{'role': 'user', 'content': 'test'}], {'type': 'object'}, 'test')
                second = client.ask('testmodel', [{'role': 'user', 'content': 'test'}], {'type': 'object'}, 'test')
                self.assertEqual(first, second)
                self.assertEqual(request.call_count, 1)

    def test_ollama_context_overflow_prevents_call(self):
        with tempfile.TemporaryDirectory() as d:
            client = Ollama('http://local', Path(d), CharacterTokenizer(), 50)
            client.models = {'testmodel': {'digest': 'abc'}}
            with patch('otse_auto.labeling.http_json') as request:
                with self.assertRaises(ValueError):
                    client.ask('testmodel', [{'role': 'user', 'content': 'test'}], {'type': 'object'}, 'test')
                self.assertEqual(request.call_count, 0)


if __name__ == '__main__':
    unittest.main()
