"""Run the frozen AiOS video adapter against a separate licensed AiOS installation."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--aios-root', type=Path, required=True)
    p.add_argument('--videos', type=Path, required=True, help='Text file listing one video path per line')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ and os.environ.get('MV_SIGN_ALLOW_LOCAL_GPU') != '1':
        raise RuntimeError('Run GPU extraction in a Slurm allocation, or explicitly set MV_SIGN_ALLOW_LOCAL_GPU=1 on a workstation')
    root = a.aios_root.resolve()
    for path in (root / 'config', root / 'datasets', a.checkpoint, a.videos):
        if not path.exists():
            raise FileNotFoundError(path)
    a.output.mkdir(parents=True, exist_ok=True)
    frozen = Path(__file__).resolve().parent / 'aios_adapter'
    with tempfile.TemporaryDirectory(prefix='aios_runtime_', dir=a.output.resolve()) as temporary:
        shadow = Path(temporary)
        for source in root.iterdir():
            if source.name in ('config', 'datasets', 'main3.py', 'engine.py', '__pycache__') or source.name.startswith(('output', 'exps', 'slurm_logs')):
                continue
            (shadow / source.name).symlink_to(source, target_is_directory=source.is_dir())
        shutil.copytree(root / 'config', shadow / 'config', ignore=shutil.ignore_patterns('__pycache__'))
        for filename in ('aios_smplx_demo.py', 'aios_smplx.py'):
            shutil.copy2(frozen / 'aios_smplx_demo.py', shadow / 'config' / filename)
        (shadow / 'datasets').mkdir()
        for source in (root / 'datasets').iterdir():
            if source.name not in ('INFERENCE_demo.py', '__pycache__'):
                (shadow / 'datasets' / source.name).symlink_to(source, target_is_directory=source.is_dir())
        shutil.copy2(frozen / 'INFERENCE_demo.py', shadow / 'datasets/INFERENCE_demo.py')
        for name in ('main3.py', 'engine.py'):
            shutil.copy2(frozen / name, shadow / name)
        subprocess.run([sys.executable, str(shadow / 'main3.py'), '-c', str(shadow / 'config/aios_smplx_demo.py'),
                        '--options', f'batch_size={a.batch_size}', 'num_person=1', 'threshold=0.1',
                        '--num_workers', str(a.workers), '--resume', str(a.checkpoint.resolve()),
                        '--eval', '--inference', '--inference_input', str(a.videos.resolve()),
                        '--output_dir', str(a.output.resolve())], cwd=shadow, check=True)


if __name__ == '__main__':
    main()
