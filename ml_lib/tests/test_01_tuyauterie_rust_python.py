import ctypes
import os

# Charger la lib Rust
lib_path = os.path.join(os.path.dirname(__file__), "../target/debug/libml_lib.dylib")
ml_lib = ctypes.CDLL(lib_path)

# Déclarer les types
ml_lib.addition.argtypes = [ctypes.c_int32, ctypes.c_int32]
ml_lib.addition.restype = ctypes.c_int32

ml_lib.soustraction.argtypes = [ctypes.c_int32, ctypes.c_int32]
ml_lib.soustraction.restype = ctypes.c_int32

# Tester
print("Test tuyauterie Rust <-> Python")
print(f"3 + 4 = {ml_lib.addition(3, 4)}")
print(f"10 - 3 = {ml_lib.soustraction(10, 3)}")
print("tuyauterie OK")