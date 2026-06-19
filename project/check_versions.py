import sys, torch, torchvision

print('=' * 45)
print('  Environment Version Check')
print('=' * 45)
print(f'Python      : {sys.version.split()[0]}')
print(f'PyTorch     : {torch.__version__}')
print(f'torchvision : {torchvision.__version__}')
print(f'CUDA avail  : {torch.cuda.is_available()}')
print(f'CUDA ver    : {torch.version.cuda}')
print(f'GPU         : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')
print()

import pkg_resources
pkgs = [
    'numpy', 'pillow', 'matplotlib', 'optuna', 'torchmetrics',
    'keras', 'keras-hub', 'tensorflow-datasets', 'onnx', 'h5py',
    'tqdm', 'torch', 'torchvision', 'torchaudio', 'rich',
    'packaging', 'requests', 'certifi',
]
print(f'{"Package":<25} {"Version"}')
print('-' * 40)
for p in sorted(pkgs):
    try:
        ver = pkg_resources.get_distribution(p).version
        print(f'{p:<25} {ver}')
    except Exception:
        print(f'{p:<25} not installed')
print('=' * 45)
