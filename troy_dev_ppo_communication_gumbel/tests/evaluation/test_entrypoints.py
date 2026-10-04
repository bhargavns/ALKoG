"""Portable entry points and experiment-launch safeguards."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


def launcher():
    spec = importlib.util.spec_from_file_location('launcher', ROOT / 'evaluation/run_entropy_pair.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('script', [
    'perception/download_sam_checkpoint', 'perception/generate_category_dataset',
    'perception/train_category_model', 'perception/evaluate_category_model',
    'communication/train_communication', 'evaluation/evaluate_communication',
    'evaluation/run_entropy_pair', 'evaluation/probe_gumbel_direct',
    'evaluation/probe_receiver_symbols', 'evaluation/compare_gumbel_vocabularies',
    'evaluation/compare_sender_entropy', 'evaluation/inspect_gumbel_failure',
])
@pytest.mark.parametrize('invocation', ['file', 'module'])
def test_help_without_external_core(script, invocation):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env.pop('CORE', None)
    command = ([str(ROOT / f'{script}.py')] if invocation == 'file'
               else ['-m', script.replace('/', '.')])
    result = subprocess.run([sys.executable, *command, '--help'], cwd=ROOT,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_unknown_training_options_are_rejected():
    from shared.ArgumentParser import ArgumentParser
    parser = ArgumentParser()
    with pytest.raises(ValueError, match='Unknown argument'):
        parser.setArguments(['misspelled=1'], [], {'iterations': '2'})
    parser.setArguments(['path=a=b'], [], {'path': ''})
    assert parser.get('path') == 'a=b'


@pytest.mark.parametrize('passed', [False, True])
@pytest.mark.parametrize('control', ['entropy', 'detached'])
def test_launcher_preflight_and_matched_commands(tmp_path, monkeypatch, passed, control):
    module = launcher()
    for folder in ('environment', 'perception', 'representation', 'communication', 'evaluation', 'shared', 'observability', 'tests'):
        (tmp_path / folder).mkdir()
    for name in ('requirements.txt', 'pytest.ini'):
        (tmp_path / name).write_text('')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0 if passed else 2))
    start = Mock()
    monkeypatch.setattr(module.subprocess, 'Popen', start)
    args = SimpleNamespace(control=control, colocated=0, iterations=2, horizon=8,
                           device='cpu', eval_episodes=2, checkpoint_every=1)
    if passed:
        module.prepare(args)
    else:
        with pytest.raises(RuntimeError, match='no arena launched'):
            module.prepare(args)
    start.assert_not_called()
    manifest = json.loads(next((tmp_path / 'output/experiments').glob('*/launch.json')).read_text())
    assert bool(manifest.get('preflight_passed')) == passed
    commands = list(manifest['commands'].values())
    options = [dict(arg.split('=', 1) for arg in command[2:]) for command in commands]
    differences = {key for key in options[0] if options[0][key] != options[1][key]}
    assert differences == {'run_name', 'sender_entropy_beta' if control == 'entropy' else 'detach_channel'}


def test_launch_rejects_changed_source(tmp_path, monkeypatch):
    module = launcher()
    (tmp_path / 'source').mkdir()
    (tmp_path / 'source/file.py').write_text('changed')
    (tmp_path / 'launch.json').write_text(json.dumps({
        'preflight_passed': True, 'source_sha256': {'file.py': 'wrong'}}))
    start = Mock()
    monkeypatch.setattr(module.subprocess, 'Popen', start)
    monkeypatch.setattr(sys, 'argv', ['launcher', '--launch', str(tmp_path)])
    with pytest.raises(RuntimeError, match='source changed'):
        module.main()
    start.assert_not_called()


@pytest.mark.parametrize('script', ['compare_gumbel_vocabularies', 'compare_sender_entropy',
                                   'inspect_gumbel_failure'])
def test_diagnostics_do_not_require_saved_outputs(tmp_path, monkeypatch, script):
    monkeypatch.syspath_prepend(str(ROOT / 'evaluation'))
    spec = importlib.util.spec_from_file_location(script, ROOT / 'evaluation' / f'{script}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_run = module.run
    # Exercise complete orchestration with short mechanism runs, not a long sweep.
    monkeypatch.setattr(module, 'run', lambda seed, **kw: original_run(seed, steps=2, **kw))
    if script == 'inspect_gumbel_failure':
        monkeypatch.setattr(module, 'audit', lambda *args: {'classes': [], 'collisions': []})
    destination = tmp_path / 'results'
    monkeypatch.setattr(sys, 'argv', [script, '--out', str(destination)])
    module.main()
    result = destination / ('audit.json' if script == 'inspect_gumbel_failure' else 'results.json')
    assert json.loads(result.read_text())['results']
