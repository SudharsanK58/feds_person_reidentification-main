import platform
import psutil
import GPUtil

def get_cpu_info():
    print("🔹 CPU Information")
    print(f"Processor: {platform.processor()}")
    print(f"Cores (Physical): {psutil.cpu_count(logical=False)}")
    print(f"Cores (Logical): {psutil.cpu_count(logical=True)}")
    print(f"CPU Frequency: {psutil.cpu_freq().max:.2f} MHz")

def get_gpu_info():
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            print("\n🔹 GPU Information")
            for gpu in gpus:
                print(f"GPU: {gpu.name}")
                print(f"  Total Memory: {gpu.memoryTotal} MB")
                print(f"  Free Memory: {gpu.memoryFree} MB")
                print(f"  Used Memory: {gpu.memoryUsed} MB")
                print(f"  GPU Load: {gpu.load * 100:.2f}%")
        else:
            print("\n🔹 GPU Information")
            print("  No GPU found.")
    except Exception as e:
        print("\n🔹 GPU Information")
        print(f"  Error detecting GPU: {e}")

if __name__ == "__main__":
    get_cpu_info()
    get_gpu_info()
