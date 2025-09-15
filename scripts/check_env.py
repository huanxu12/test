#!/usr/bin/env python3
"""
MineSLAM Environment Checker
Verifies CUDA, drivers, and MinkowskiEngine availability for real data processing
"""

import sys
import subprocess
import importlib
from typing import Dict, List, Tuple
import platform
import psutil


def check_python_version() -> Tuple[bool, str]:
    """Check Python version compatibility"""
    version = sys.version_info
    required_major, required_minor = 3, 8
    
    is_compatible = version.major >= required_major and version.minor >= required_minor
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    return is_compatible, version_str


def check_cuda_availability() -> Dict[str, any]:
    """Check CUDA installation and availability"""
    cuda_info = {
        'available': False,
        'version': None,
        'devices': [],
        'driver_version': None,
        'memory_info': {}
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            cuda_info['available'] = True
            cuda_info['version'] = torch.version.cuda
            cuda_info['device_count'] = torch.cuda.device_count()
            
            # Get device information
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                device_info = {
                    'name': device_props.name,
                    'total_memory': device_props.total_memory,
                    'major': device_props.major,
                    'minor': device_props.minor,
                    'multi_processor_count': device_props.multi_processor_count
                }
                cuda_info['devices'].append(device_info)
                
                # Memory info for current device
                if i == 0:  # Primary device
                    cuda_info['memory_info'] = {
                        'total': torch.cuda.get_device_properties(0).total_memory,
                        'allocated': torch.cuda.memory_allocated(0),
                        'reserved': torch.cuda.memory_reserved(0)
                    }
        
        # Try to get driver version from nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, check=True)
            cuda_info['driver_version'] = result.stdout.strip().split('\n')[0]
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
            
    except ImportError:
        pass
    
    return cuda_info


def check_pytorch_installation() -> Dict[str, any]:
    """Check PyTorch installation and configuration"""
    pytorch_info = {
        'installed': False,
        'version': None,
        'cuda_version': None,
        'cudnn_version': None,
        'cuda_available': False,
        'mps_available': False
    }
    
    try:
        import torch
        pytorch_info['installed'] = True
        pytorch_info['version'] = torch.__version__
        pytorch_info['cuda_available'] = torch.cuda.is_available()
        
        if hasattr(torch.backends, 'mps'):
            pytorch_info['mps_available'] = torch.backends.mps.is_available()
        
        if torch.cuda.is_available():
            pytorch_info['cuda_version'] = torch.version.cuda
            
        if hasattr(torch.backends, 'cudnn'):
            pytorch_info['cudnn_version'] = torch.backends.cudnn.version()
            
    except ImportError:
        pass
    
    return pytorch_info


def check_minkowski_engine() -> Dict[str, any]:
    """Check MinkowskiEngine installation for point cloud processing"""
    minkowski_info = {
        'installed': False,
        'version': None,
        'cuda_support': False,
        'import_error': None
    }
    
    try:
        import MinkowskiEngine as ME
        minkowski_info['installed'] = True
        
        if hasattr(ME, '__version__'):
            minkowski_info['version'] = ME.__version__
        
        # Test CUDA support by creating a simple sparse tensor
        try:
            import torch
            if torch.cuda.is_available():
                coords = torch.IntTensor([[0, 0, 0, 0], [0, 0, 0, 1]])
                feats = torch.FloatTensor([[1], [1]])
                
                sparse_tensor = ME.SparseTensor(
                    features=feats,
                    coordinates=coords,
                    device='cuda'
                )
                minkowski_info['cuda_support'] = True
        except Exception as e:
            minkowski_info['cuda_support'] = False
            minkowski_info['cuda_error'] = str(e)
            
    except ImportError as e:
        minkowski_info['import_error'] = str(e)
    
    return minkowski_info


def check_required_packages() -> Dict[str, Dict[str, any]]:
    """Check all required packages for MineSLAM"""
    required_packages = [
        'torch',
        'torchvision', 
        'numpy',
        'opencv-python',
        'pillow',
        'scipy',
        'tensorboard',
        'yaml',
        'tqdm'
    ]
    
    optional_packages = [
        'MinkowskiEngine',
        'open3d',
        'matplotlib',
        'seaborn'
    ]
    
    package_info = {}
    
    for package in required_packages + optional_packages:
        info = {
            'installed': False,
            'version': None,
            'required': package in required_packages
        }
        
        try:
            # Handle package name differences
            import_name = package
            if package == 'opencv-python':
                import_name = 'cv2'
            elif package == 'pillow':
                import_name = 'PIL'
            elif package == 'yaml':
                import_name = 'yaml'
            
            module = importlib.import_module(import_name)
            info['installed'] = True
            
            if hasattr(module, '__version__'):
                info['version'] = module.__version__
            elif hasattr(module, 'version'):
                info['version'] = module.version
                
        except ImportError:
            pass
        
        package_info[package] = info
    
    return package_info


def check_system_resources() -> Dict[str, any]:
    """Check system resources and capabilities"""
    system_info = {
        'platform': platform.platform(),
        'architecture': platform.architecture()[0],
        'processor': platform.processor(),
        'cpu_count': psutil.cpu_count(logical=True),
        'cpu_count_physical': psutil.cpu_count(logical=False),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_usage': {}
    }
    
    # Check disk usage for common mount points
    import os
    for path in ['/', '/tmp', '/home']:
        if os.path.exists(path):
            usage = psutil.disk_usage(path)
            system_info['disk_usage'][path] = {
                'total': usage.total,
                'free': usage.free,
                'used': usage.used
            }
    
    return system_info


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def print_environment_report():
    """Print comprehensive environment report"""
    print("="*80)
    print("MineSLAM ENVIRONMENT CHECK - REAL DATA PROCESSING REQUIREMENTS")
    print("="*80)
    
    # Python version
    python_ok, python_version = check_python_version()
    status = "✓" if python_ok else "✗"
    print(f"\n{status} Python Version: {python_version}")
    if not python_ok:
        print("  ⚠ Python 3.8+ required for MineSLAM")
    
    # System resources
    print(f"\n📊 SYSTEM RESOURCES:")
    system_info = check_system_resources()
    print(f"  Platform: {system_info['platform']}")
    print(f"  CPU: {system_info['processor']} ({system_info['cpu_count']} cores)")
    print(f"  Memory: {format_bytes(system_info['memory_total'])} total, {format_bytes(system_info['memory_available'])} available")
    
    # CUDA check
    print(f"\n🚀 CUDA SUPPORT:")
    cuda_info = check_cuda_availability()
    if cuda_info['available']:
        print(f"  ✓ CUDA Available: {cuda_info['version']}")
        print(f"  ✓ Driver Version: {cuda_info.get('driver_version', 'Unknown')}")
        print(f"  ✓ GPU Devices: {cuda_info.get('device_count', 0)}")
        
        for i, device in enumerate(cuda_info['devices']):
            memory_gb = device['total_memory'] / (1024**3)
            print(f"    GPU {i}: {device['name']} ({memory_gb:.1f} GB)")
            
        if cuda_info['memory_info']:
            total_gb = cuda_info['memory_info']['total'] / (1024**3)
            allocated_gb = cuda_info['memory_info']['allocated'] / (1024**3)
            print(f"  Memory: {allocated_gb:.1f}/{total_gb:.1f} GB allocated")
    else:
        print("  ✗ CUDA Not Available")
        print("  ⚠ GPU acceleration required for real-time processing")
    
    # PyTorch check
    print(f"\n🔥 PYTORCH:")
    pytorch_info = check_pytorch_installation()
    if pytorch_info['installed']:
        print(f"  ✓ PyTorch: {pytorch_info['version']}")
        print(f"  ✓ CUDA Support: {'Yes' if pytorch_info['cuda_available'] else 'No'}")
        if pytorch_info['cudnn_version']:
            print(f"  ✓ cuDNN Version: {pytorch_info['cudnn_version']}")
    else:
        print("  ✗ PyTorch Not Installed")
        print("  ⚠ PyTorch required for MineSLAM")
    
    # MinkowskiEngine check
    print(f"\n📦 MINKOWSKI ENGINE (Point Cloud Processing):")
    minkowski_info = check_minkowski_engine()
    if minkowski_info['installed']:
        print(f"  ✓ MinkowskiEngine: {minkowski_info.get('version', 'Unknown version')}")
        print(f"  ✓ CUDA Support: {'Yes' if minkowski_info['cuda_support'] else 'No'}")
        if not minkowski_info['cuda_support'] and 'cuda_error' in minkowski_info:
            print(f"    Error: {minkowski_info['cuda_error']}")
    else:
        print("  ✗ MinkowskiEngine Not Installed")
        print("  ⚠ MinkowskiEngine required for point cloud processing")
        if minkowski_info['import_error']:
            print(f"    Error: {minkowski_info['import_error']}")
    
    # Package dependencies
    print(f"\n📚 PACKAGE DEPENDENCIES:")
    package_info = check_required_packages()
    
    required_missing = []
    optional_missing = []
    
    for package, info in package_info.items():
        status = "✓" if info['installed'] else "✗"
        version_str = f" ({info['version']})" if info['version'] else ""
        package_type = "Required" if info['required'] else "Optional"
        
        print(f"  {status} {package}{version_str} - {package_type}")
        
        if not info['installed']:
            if info['required']:
                required_missing.append(package)
            else:
                optional_missing.append(package)
    
    # Summary and recommendations
    print(f"\n🎯 ENVIRONMENT SUMMARY:")
    
    critical_issues = []
    warnings = []
    
    if not python_ok:
        critical_issues.append("Python version too old")
    
    if not pytorch_info['installed']:
        critical_issues.append("PyTorch not installed")
    
    if not cuda_info['available']:
        warnings.append("CUDA not available - will use CPU (slow)")
    
    if not minkowski_info['installed']:
        warnings.append("MinkowskiEngine not available - point cloud processing limited")
    
    if required_missing:
        critical_issues.append(f"Missing required packages: {', '.join(required_missing)}")
    
    if critical_issues:
        print("  ❌ CRITICAL ISSUES:")
        for issue in critical_issues:
            print(f"    • {issue}")
        print("\n  🔧 INSTALL COMMANDS:")
        if required_missing:
            print(f"    pip install {' '.join(required_missing)}")
        if not minkowski_info['installed']:
            print("    # MinkowskiEngine installation:")
            print("    pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option='--blas_include_dirs=${CONDA_PREFIX}/include' --install-option='--blas=openblas'")
    else:
        print("  ✅ Environment ready for MineSLAM training!")
    
    if warnings:
        print("  ⚠ WARNINGS:")
        for warning in warnings:
            print(f"    • {warning}")
    
    # Performance recommendations
    if cuda_info['available']:
        total_gpu_memory = sum(device['total_memory'] for device in cuda_info['devices'])
        total_gpu_gb = total_gpu_memory / (1024**3)
        
        if total_gpu_gb < 8:
            print(f"  💡 PERFORMANCE NOTE: GPU memory ({total_gpu_gb:.1f} GB) may limit batch size")
        elif total_gpu_gb >= 16:
            print(f"  🚀 PERFORMANCE: Excellent GPU memory ({total_gpu_gb:.1f} GB) for large batches")
    
    print("="*80)


def main():
    """Run environment check"""
    try:
        print_environment_report()
        return 0
    except Exception as e:
        print(f"Error during environment check: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())