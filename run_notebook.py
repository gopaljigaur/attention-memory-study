"""Execute the walkthrough in the invoking Python environment, save real outputs."""
from pathlib import Path
import json
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

root = Path(__file__).resolve().parent
kernel_dir = root / '.jupyter/kernels/attention'
kernel_dir.mkdir(parents=True, exist_ok=True)
(kernel_dir / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Attention study', 'language': 'python',
}))
manager = KernelManager(kernel_name='attention', kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)]))
notebook = nbformat.read(root / 'attention-memory.ipynb', as_version=4)
NotebookClient(notebook, km=manager, timeout=180, resources={'metadata': {'path': str(root)}}).execute()
nbformat.validate(notebook)
assert not any(output.output_type == 'error' for cell in notebook.cells if cell.cell_type == 'code' for output in cell.get('outputs', []))
nbformat.write(notebook, root / 'attention-memory.ipynb')
print('Executed and validated all notebook cells without errors.')
