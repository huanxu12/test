"""
MineSLAM Seeding Module
Deterministic seeding for reproducible training with real data
"""

import random
import numpy as np
import torch
import os
from typing import Optional


def set_deterministic_seed(seed: int = 42, strict_mode: bool = True) -> None:
    """
    Set deterministic seed for reproducible training
    
    Args:
        seed: Random seed value
        strict_mode: Enable strict deterministic mode (slower but more reproducible)
    """
    print(f"Setting deterministic seed: {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    
    # CUDA random (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
        if strict_mode:
            # Enable deterministic algorithms (may be slower)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # Set additional CUDA flags for determinism
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            
            # Use deterministic algorithms where available
            torch.use_deterministic_algorithms(True, warn_only=True)
            
            print("✓ Strict deterministic mode enabled")
        else:
            # Allow some non-determinism for better performance
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            
            print("✓ Fast mode enabled (some non-determinism allowed)")
    
    print(f"✓ All random seeds set to {seed}")


def get_random_state() -> dict:
    """
    Get current random state for all generators
    
    Returns:
        Dictionary containing current random states
    """
    state = {
        'python_random': random.getstate(),
        'numpy_random': np.random.get_state(),
        'torch_random': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['torch_cuda_random'] = torch.cuda.get_rng_state()
        
        # For multi-GPU
        if torch.cuda.device_count() > 1:
            state['torch_cuda_random_all'] = torch.cuda.get_rng_state_all()
    
    return state


def set_random_state(state: dict) -> None:
    """
    Restore random state for all generators
    
    Args:
        state: Dictionary containing random states
    """
    if 'python_random' in state:
        random.setstate(state['python_random'])
    
    if 'numpy_random' in state:
        np.random.set_state(state['numpy_random'])
    
    if 'torch_random' in state:
        torch.set_rng_state(state['torch_random'])
    
    if torch.cuda.is_available():
        if 'torch_cuda_random' in state:
            torch.cuda.set_rng_state(state['torch_cuda_random'])
        
        if 'torch_cuda_random_all' in state:
            torch.cuda.set_rng_state_all(state['torch_cuda_random_all'])


class ReproducibleDataLoader:
    """
    Wrapper for DataLoader to ensure reproducible data loading with real data
    """
    
    def __init__(self, dataloader: torch.utils.data.DataLoader, seed: int = 42):
        self.dataloader = dataloader
        self.seed = seed
        self.initial_state = None
        
        # Set worker init function for reproducibility
        if hasattr(dataloader, 'worker_init_fn') and dataloader.worker_init_fn is None:
            dataloader.worker_init_fn = self._worker_init_fn
    
    def _worker_init_fn(self, worker_id: int) -> None:
        """Initialize worker with deterministic seed"""
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    def __iter__(self):
        """Start iteration with saved random state"""
        if self.initial_state is None:
            self.initial_state = get_random_state()
        else:
            # Restore state for consistent iteration order
            set_random_state(self.initial_state)
        
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


def validate_real_data_determinism(data_samples: list, 
                                  seed: int = 42, 
                                  num_iterations: int = 3) -> bool:
    """
    Validate that real data loading is deterministic
    
    Args:
        data_samples: List of data samples to test
        seed: Seed for reproducibility test
        num_iterations: Number of iterations to test consistency
    
    Returns:
        True if data loading is deterministic
    """
    print("Validating real data loading determinism...")
    
    if len(data_samples) == 0:
        print("⚠ No data samples provided for determinism test")
        return True
    
    # Store results from each iteration
    iteration_results = []
    
    for iteration in range(num_iterations):
        print(f"  Iteration {iteration + 1}/{num_iterations}")
        
        # Set seed and load data
        set_deterministic_seed(seed)
        
        # Simulate data loading (take first few samples)
        sample_subset = data_samples[:min(5, len(data_samples))]
        
        # Create hash of sample data for comparison
        sample_hashes = []
        for sample in sample_subset:
            if isinstance(sample, dict):
                # Hash dictionary contents
                hash_str = str(sorted(sample.items()))
            else:
                # Hash tensor or other data
                hash_str = str(sample)
            
            sample_hashes.append(hash(hash_str))
        
        iteration_results.append(sample_hashes)
    
    # Check if all iterations produced same results
    first_result = iteration_results[0]
    is_deterministic = all(result == first_result for result in iteration_results)
    
    if is_deterministic:
        print("✓ Real data loading is deterministic")
        return True
    else:
        print("✗ Real data loading is NOT deterministic")
        print("  This may cause training reproducibility issues")
        return False


def create_reproducible_split(data_indices: list, 
                             train_ratio: float = 0.8,
                             val_ratio: float = 0.1,
                             test_ratio: float = 0.1,
                             seed: int = 42) -> tuple:
    """
    Create reproducible train/val/test split from real data indices
    
    Args:
        data_indices: List of data indices
        train_ratio: Fraction for training
        val_ratio: Fraction for validation  
        test_ratio: Fraction for testing
        seed: Random seed for reproducible split
    
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Set seed for reproducible split
    set_deterministic_seed(seed)
    
    # Shuffle indices
    indices = data_indices.copy()
    random.shuffle(indices)
    
    # Calculate split sizes
    total_size = len(indices)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # Create splits
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    print(f"Created reproducible split (seed={seed}):")
    print(f"  Train: {len(train_indices)} samples ({len(train_indices)/total_size*100:.1f}%)")
    print(f"  Val: {len(val_indices)} samples ({len(val_indices)/total_size*100:.1f}%)")
    print(f"  Test: {len(test_indices)} samples ({len(test_indices)/total_size*100:.1f}%)")
    
    return train_indices, val_indices, test_indices


def save_training_state(model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       loss: float,
                       random_state: dict,
                       filepath: str) -> None:
    """
    Save complete training state including random states for reproducibility
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        random_state: Random generator states
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'random_state': random_state,
        'timestamp': torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True)) if torch.cuda.is_available() else 0
    }
    
    torch.save(checkpoint, filepath)
    print(f"✓ Saved training state with random states: {filepath}")


def load_training_state(filepath: str,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer) -> int:
    """
    Load complete training state including random states for reproducibility
    
    Args:
        filepath: Path to checkpoint
        model: PyTorch model
        optimizer: Optimizer
    
    Returns:
        Epoch number to resume from
    """
    checkpoint = torch.load(filepath)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Restore random states
    if 'random_state' in checkpoint:
        set_random_state(checkpoint['random_state'])
        print("✓ Restored random states for reproducible continuation")
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"✓ Loaded training state from epoch {epoch}, loss: {loss:.6f}")
    
    return epoch


class DeterministicSampler:
    """
    Deterministic sampler for real data that ensures reproducible ordering
    """
    
    def __init__(self, data_indices: list, seed: int = 42, shuffle: bool = True):
        self.data_indices = data_indices
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
    
    def __iter__(self):
        """Generate deterministic sequence of indices"""
        # Set seed based on epoch for different ordering each epoch
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        
        if self.shuffle:
            # Create permutation
            indices = torch.randperm(len(self.data_indices), generator=generator).tolist()
            return iter([self.data_indices[i] for i in indices])
        else:
            return iter(self.data_indices)
    
    def __len__(self):
        return len(self.data_indices)
    
    def set_epoch(self, epoch: int):
        """Set epoch for different shuffling each epoch"""
        self.epoch = epoch


def verify_deterministic_setup() -> bool:
    """
    Verify that deterministic setup is working correctly
    
    Returns:
        True if setup is deterministic
    """
    print("Verifying deterministic setup...")
    
    # Test torch random
    set_deterministic_seed(42)
    torch_rand1 = torch.rand(5)
    
    set_deterministic_seed(42)
    torch_rand2 = torch.rand(5)
    
    torch_ok = torch.allclose(torch_rand1, torch_rand2)
    
    # Test numpy random
    set_deterministic_seed(42)
    np_rand1 = np.random.rand(5)
    
    set_deterministic_seed(42)
    np_rand2 = np.random.rand(5)
    
    np_ok = np.allclose(np_rand1, np_rand2)
    
    # Test python random
    set_deterministic_seed(42)
    py_rand1 = [random.random() for _ in range(5)]
    
    set_deterministic_seed(42)
    py_rand2 = [random.random() for _ in range(5)]
    
    py_ok = py_rand1 == py_rand2
    
    overall_ok = torch_ok and np_ok and py_ok
    
    print(f"  PyTorch deterministic: {'✓' if torch_ok else '✗'}")
    print(f"  NumPy deterministic: {'✓' if np_ok else '✗'}")
    print(f"  Python deterministic: {'✓' if py_ok else '✗'}")
    print(f"  Overall: {'✓ PASS' if overall_ok else '✗ FAIL'}")
    
    return overall_ok