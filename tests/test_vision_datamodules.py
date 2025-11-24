#!/usr/bin/env python
"""
Test script for vision DataModules.
Verifies that MNIST and CIFAR DataModules work with sequentialization.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_vision_imports():
    """Test that vision DataModules can be imported."""
    print("Testing vision DataModule imports...")
    try:
        from elmneuron.datasets import (
            CIFAR10SequenceDataModule,
            CIFAR100SequenceDataModule,
            FashionMNISTSequenceDataModule,
            MNISTSequenceDataModule,
        )
        from elmneuron.transforms import PatchSequence, PixelSequence

        print("✓ All vision DataModules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("  Note: Vision DataModules require torchvision")
        print("  Install with: pip install elmneuron[vision]")
        return False


def test_mnist_creation():
    """Test MNIST DataModule creation."""
    print("\nTesting MNIST DataModule creation...")
    try:
        from elmneuron.datasets import MNISTSequenceDataModule
        from elmneuron.transforms import PatchSequence

        # Create with patch sequentialization
        datamodule = MNISTSequenceDataModule(
            data_dir="/tmp/test_mnist",
            sequence_transform=PatchSequence(patch_size=7, order="raster"),
            batch_size=32,
        )

        info = datamodule.get_dataset_info()
        print(f"✓ MNIST DataModule created successfully")
        print(f"  Num classes: {info['num_classes']}")
        print(f"  Image shape: {info['image_shape']}")
        print(f"  Sequence length: {info['sequence_length']}")
        print(f"  Input dim per timestep: {info['input_dim']}")

        return True
    except Exception as e:
        print(f"✗ Creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_cifar10_creation():
    """Test CIFAR-10 DataModule creation."""
    print("\nTesting CIFAR-10 DataModule creation...")
    try:
        from elmneuron.datasets import CIFAR10SequenceDataModule
        from elmneuron.transforms import PatchSequence

        # Create with patch sequentialization
        datamodule = CIFAR10SequenceDataModule(
            data_dir="/tmp/test_cifar",
            sequence_transform=PatchSequence(patch_size=8, order="raster"),
            batch_size=32,
            data_augmentation=True,
        )

        info = datamodule.get_dataset_info()
        print(f"✓ CIFAR-10 DataModule created successfully")
        print(f"  Num classes: {info['num_classes']}")
        print(f"  Image shape: {info['image_shape']}")
        print(f"  Sequence length: {info['sequence_length']}")
        print(f"  Input dim per timestep: {info['input_dim']}")

        return True
    except Exception as e:
        print(f"✗ Creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_pixel_sequentialization():
    """Test pixel-based sequentialization."""
    print("\nTesting pixel sequentialization...")
    try:
        from elmneuron.datasets import MNISTSequenceDataModule
        from elmneuron.transforms import PixelSequence

        # Create with pixel sequentialization
        datamodule = MNISTSequenceDataModule(
            data_dir="/tmp/test_mnist",
            sequence_transform=PixelSequence(order="raster", flatten_channels=True),
            batch_size=32,
        )

        info = datamodule.get_dataset_info()
        print(f"✓ Pixel sequentialization works")
        print(
            f"  Sequence length: {info['sequence_length']}"
        )  # Should be 784 for MNIST
        print(
            f"  Input dim per timestep: {info['input_dim']}"
        )  # Should be 1 (grayscale)

        expected_seq_len = 28 * 28  # MNIST is 28x28
        assert (
            info["sequence_length"] == expected_seq_len
        ), f"Expected sequence length {expected_seq_len}, got {info['sequence_length']}"

        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_multiple_patch_sizes():
    """Test different patch sizes."""
    print("\nTesting multiple patch sizes...")
    try:
        from elmneuron.datasets import MNISTSequenceDataModule
        from elmneuron.transforms import PatchSequence

        patch_sizes = [7, 14]  # 28 is divisible by both
        for patch_size in patch_sizes:
            datamodule = MNISTSequenceDataModule(
                data_dir="/tmp/test_mnist",
                sequence_transform=PatchSequence(patch_size=patch_size),
                batch_size=32,
            )

            info = datamodule.get_dataset_info()
            expected_patches = (28 // patch_size) ** 2

            print(f"✓ Patch size {patch_size}x{patch_size}:")
            print(
                f"    Sequence length: {info['sequence_length']} (expected: {expected_patches})"
            )
            print(f"    Input dim: {info['input_dim']}")

            assert (
                info["sequence_length"] == expected_patches
            ), f"Expected {expected_patches} patches, got {info['sequence_length']}"

        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_fashion_mnist():
    """Test Fashion-MNIST DataModule."""
    print("\nTesting Fashion-MNIST DataModule...")
    try:
        from elmneuron.datasets import FashionMNISTSequenceDataModule

        datamodule = FashionMNISTSequenceDataModule(
            data_dir="/tmp/test_fashion_mnist",
            batch_size=32,
        )

        info = datamodule.get_dataset_info()
        print(f"✓ Fashion-MNIST DataModule created successfully")
        print(f"  Num classes: {info['num_classes']}")
        print(f"  Sequence length: {info['sequence_length']}")

        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_cifar100():
    """Test CIFAR-100 DataModule."""
    print("\nTesting CIFAR-100 DataModule...")
    try:
        from elmneuron.datasets import CIFAR100SequenceDataModule

        datamodule = CIFAR100SequenceDataModule(
            data_dir="/tmp/test_cifar100",
            batch_size=32,
        )

        info = datamodule.get_dataset_info()
        print(f"✓ CIFAR-100 DataModule created successfully")
        print(f"  Num classes: {info['num_classes']}")
        assert info["num_classes"] == 100, "CIFAR-100 should have 100 classes"

        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sequential_wrapper():
    """Test sequential dataset wrapper."""
    print("\nTesting sequential dataset wrapper...")
    try:
        import torch
        from torch.utils.data import TensorDataset

        from elmneuron.datasets.vision_datamodule import SequentialDatasetWrapper
        from elmneuron.transforms import PatchSequence

        # Create dummy dataset
        images = torch.randn(10, 1, 28, 28)  # 10 MNIST-like images
        labels = torch.randint(0, 10, (10,))
        base_dataset = TensorDataset(images, labels)

        # Wrap with sequentialization
        transform = PatchSequence(patch_size=7)
        wrapped = SequentialDatasetWrapper(base_dataset, transform, normalize=False)

        # Test __getitem__
        sequence, label = wrapped[0]
        print(f"✓ Sequential wrapper works")
        print(f"  Sequence shape: {sequence.shape}")
        print(f"  Label: {label}")

        # Should be (num_patches, patch_features)
        expected_patches = (28 // 7) ** 2  # 16 patches
        assert (
            sequence.shape[0] == expected_patches
        ), f"Expected {expected_patches} patches, got {sequence.shape[0]}"

        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Vision DataModules")
    print("=" * 60)

    results = []

    # Basic tests
    results.append(("Vision Imports", test_vision_imports()))

    # Only run other tests if imports succeeded
    if results[0][1]:
        results.append(("MNIST Creation", test_mnist_creation()))
        results.append(("CIFAR-10 Creation", test_cifar10_creation()))
        results.append(("Pixel Sequentialization", test_pixel_sequentialization()))
        results.append(("Multiple Patch Sizes", test_multiple_patch_sizes()))
        results.append(("Fashion-MNIST", test_fashion_mnist()))
        results.append(("CIFAR-100", test_cifar100()))
        results.append(("Sequential Wrapper", test_sequential_wrapper()))
    else:
        print("\n⚠ Skipping remaining tests due to import failure")
        print("  Install torchvision with: pip install torchvision pillow")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<40} {status}")

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print(f"✗ {total - passed} test(s) failed")
        if not results[0][1]:
            print(
                "  Hint: Install vision dependencies with: pip install elmneuron[vision]"
            )
        sys.exit(1)
