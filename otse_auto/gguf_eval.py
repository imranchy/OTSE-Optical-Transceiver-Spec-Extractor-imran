"""Optional matched Q4_K_M development evaluation using an existing llama-server."""
from __future__ import annotations
import shutil
import socket
import subprocess
import time
from pathlib import Path
from .core import BASE_GGUF, digest_obj, properties, read_json, read_jsonl, sha256, write_json
from .evaluation import parse_object, compare
from .labeling import http_json
from .export import find_tools


def server_binary(root, explicit=None):
    tools = find_tools(root, explicit)
    folders = [tools[0].parent] if tools else []
    for folder in folders:
        for rel in ('build/bin/Release/llama-server.exe', 'build/bin/llama-server.exe',
                    'build/bin/llama-server', 'llama-server.exe', 'llama-server'):
            path = folder / rel
            if path.is_file():
                return path
    on_path = shutil.which('llama-server') or shutil.which('llama-server.exe')
    return Path(on_path) if on_path else None


def evaluate(root, out, explicit=None):
    binary = server_binary(root, explicit)
    if binary is None:
        raise FileNotFoundError('Existing llama-server not located. Supply --llama-cpp-dir; no new binary is downloaded.')
    rows = read_jsonl(out / 'dataset/dev.jsonl')
    if not rows:
        raise FileNotFoundError('No automated dev dataset')
    sources = {'gguf_baseline': root / BASE_GGUF, 'gguf_candidate': out / 'candidate-Q4_K_M.gguf'}
    for which, gguf in sources.items():
        if not gguf.is_file():
            raise FileNotFoundError(str(gguf))
        fingerprint = sha256(gguf)
        settings = {'server_sha256': sha256(binary), 'runtime': 'llama.cpp_Q4_K_M', 'model_sha256': fingerprint, 'temperature': 0,
                    'max_tokens': 1024, 'context': 8192, 'constrained_json': True, 'gpu_layers': 99}
        pending = []
        folder = out / 'predictions' / which
        for row in rows:
            path = folder / (row['document_id'] + '__' + row['field_group'] + '.json')
            key = digest_obj([settings, row['source_view_sha256']])
            if not path.is_file() or read_json(path).get('prediction_key') != key:
                pending.append((row, path, key))
        if not pending:
            continue
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        url = f'http://127.0.0.1:{port}'
        logs = out / 'reports'
        logs.mkdir(parents=True, exist_ok=True)
        command = [str(binary), '-m', str(gguf), '-c', '8192', '-ngl', '99', '--parallel', '1',
                   '--host', '127.0.0.1', '--port', str(port), '--alias', 'otse-auto']
        with (logs / (which + '_server.log')).open('ab') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                for _ in range(180):
                    if process.poll() is not None:
                        raise RuntimeError('llama-server exited; see ' + str(log.name))
                    try:
                        health = http_json(url + '/health', timeout=2)
                        if health.get('status') == 'ok':
                            break
                    except (OSError, RuntimeError):
                        pass
                    time.sleep(1)
                else:
                    raise RuntimeError('llama-server startup timeout')
                for index, (row, path, key) in enumerate(pending, 1):
                    defs = properties(row['field_group'])
                    schema = {'type': 'object', 'properties': defs, 'required': list(defs), 'additionalProperties': False}
                    request = {'model': 'otse-auto', 'messages': row['messages'][:-1], 'temperature': 0,
                        'seed': 42, 'max_tokens': 1024, 'stream': False,
                        'response_format': {'type': 'json_schema', 'json_schema': {'name': 'otse', 'strict': True, 'schema': schema}}}
                    start = time.perf_counter()
                    response = http_json(url + '/v1/chat/completions', request, timeout=900)
                    choice = response['choices'][0]
                    content = choice.get('message', {}).get('content') or ''
                    parsed = parse_object(content)
                    write_json(path, {'document_id': row['document_id'], 'field_group': row['field_group'], 'split': 'dev',
                        'prediction_key': key, 'settings': settings, 'source_view_sha256': row['source_view_sha256'],
                        'content': content, 'parsed': parsed, 'schema_keys_ok': isinstance(parsed, dict) and set(parsed) == set(defs),
                        'possible_length_cutoff': choice.get('finish_reason') == 'length',
                        'request_wall_seconds': time.perf_counter() - start, 'usage': response.get('usage')})
                    print(f'{which}: {index}/{len(pending)} dev groups', flush=True)
            finally:
                # Only stop the server process launched by this function.
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
    return compare(out, prefix='gguf_')
