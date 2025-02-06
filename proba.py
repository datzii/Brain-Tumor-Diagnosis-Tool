import numba.cuda as cuda

try:
    device = cuda.get_current_device()
    print(f"CUDA Device: {device.name()}")
except Exception as e:
    print(f"Error: {e}")
