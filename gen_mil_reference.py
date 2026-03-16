"""Generate a simple CoreML model and extract its MIL format for macOS 26."""
import coremltools as ct
import numpy as np
import os
import subprocess
import glob

# Create simple model via torch
try:
    import torch
    import torch.nn as nn

    class SimpleAdd(nn.Module):
        def forward(self, x):
            return x + x

    model = SimpleAdd()
    model.eval()
    example = torch.randn(1, 64, 1, 64)
    traced = torch.jit.trace(model, example)
    ct_model = ct.convert(traced, inputs=[ct.TensorType(shape=(1, 64, 1, 64))],
                          minimum_deployment_target=ct.target.macOS15)
    ct_model.save("/tmp/ane_simple.mlpackage")
    print("Saved simple model")
except ImportError:
    print("No torch, trying direct approach...")
    # Use coremltools builder directly
    from coremltools.converters.mil.mil import Builder as mb

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 64, 1, 64))])
    def prog(x):
        return mb.add(x=x, y=x, name="out")

    ct_model = ct.convert(prog, minimum_deployment_target=ct.target.macOS15)
    ct_model.save("/tmp/ane_simple.mlpackage")
    print("Saved simple model")

# Compile with xcrun
print("\n--- Compiling ---")
result = subprocess.run(
    ["xcrun", "coremlcompiler", "compile", "/tmp/ane_simple.mlpackage", "/tmp/"],
    capture_output=True, text=True)
print(f"compile rc: {result.returncode}")
if result.stdout: print(f"stdout: {result.stdout[:300]}")
if result.stderr: print(f"stderr: {result.stderr[:300]}")

# Find compiled model
for pattern in ["/tmp/ane_simple.mlmodelc", "/tmp/ane_simple*"]:
    for p in glob.glob(pattern):
        print(f"\nFound: {p}")
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                for f in sorted(files):
                    path = os.path.join(root, f)
                    size = os.path.getsize(path)
                    print(f"  {os.path.relpath(path, p)}: {size} bytes")
                    if f.endswith('.mil') or f == 'model.mil':
                        with open(path, 'r', errors='replace') as fh:
                            content = fh.read()
                        print(f"  --- MIL CONTENT ({len(content)} bytes) ---")
                        for i, line in enumerate(content.split('\n')[:50], 1):
                            print(f"  {i:3}: {line}")
                        if len(content.split('\n')) > 50:
                            print(f"  ... ({len(content.split(chr(10)))} total lines)")
                        print(f"  --- END MIL ---")
