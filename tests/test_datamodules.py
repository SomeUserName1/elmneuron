#!/usr/bin/env python
"""
Test script for Lightning DataModules.
Verifies that NeuronIO and SHD DataModules can be instantiated and configured.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_neuronio_datamodule_import():
    """Test that NeuronIODataModule can be imported."""
    print("Testing NeuronIODataModule import...")
    try:
        from elmneuron.neuronio import NeuronIODataModule
        from elmneuron.transforms import NeuronIORouting

        print("✓ NeuronIODataModule imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_neuronio_datamodule_creation():
    """Test that NeuronIODataModule can be created."""
    print("\nTesting NeuronIODataModule creation...")
    try:
        from elmneuron.neuronio import NeuronIODataModule
        from elmneuron.transforms import NeuronIORouting

        # Create routing
        routing = NeuronIORouting(
            num_input=1278,
            num_branch=45,
            num_synapse_per_branch=100,
        )

        # Create DataModule (without actual files)
        datamodule = NeuronIODataModule(
            train_files=["dummy_file_1.pkl"],  # Dummy files
            val_files=["dummy_file_2.pkl"],
            routing=routing,
            batch_size=8,
            train_batches_per_epoch=10,
            val_batches_per_epoch=5,
        )

        print(f"✓ DataModule created successfully")
        print(f"  Input dimension: {datamodule.input_dim}")
        print(f"  Num classes: {datamodule.num_classes}")
        print(f"  Routing info: {datamodule.get_routing_info()}")

        return True
    except Exception as e:
        print(f"✗ Creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_neuronio_routing():
    """Test routing transforms."""
    print("\nTesting NeuronIO routing...")
    try:
        import torch

        from elmneuron.transforms import NeuronIORouting

        # Create routing
        routing = NeuronIORouting(
            num_input=1278,
            num_branch=45,
            num_synapse_per_branch=100,
        )

        # Test routing
        x = torch.randn(2, 10, 1278)  # (batch, time, input)
        x_routed = routing(x)

        expected_shape = (2, 10, 4500)  # 45 * 100
        assert (
            x_routed.shape == expected_shape
        ), f"Expected {expected_shape}, got {x_routed.shape}"

        print(f"✓ Routing works correctly")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {x_routed.shape}")
        print(f"  Routing info: {routing.get_routing_info()}")

        return True
    except Exception as e:
        print(f"✗ Routing test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_shd_datamodule_import():
    """Test that SHDDataModule can be imported."""
    print("\nTesting SHDDataModule import...")
    try:
        from elmneuron.shd import SHDDataModule

        print("✓ SHDDataModule imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_shd_datamodule_creation():
    """Test that SHDDataModule can be created."""
    print("\nTesting SHDDataModule creation...")
    try:
        from elmneuron.shd import SHDDataModule

        # Create DataModule (without downloading)
        datamodule = SHDDataModule(
            data_dir="/tmp/test_shd_cache",
            dataset_variant="shd",
            batch_size=256,
            bin_size=25,
        )

        print(f"✓ DataModule created successfully")
        print(f"  Dataset variant: {datamodule.dataset_variant}")
        print(f"  Num classes: {datamodule.num_classes}")
        print(f"  Input dim: {datamodule.input_dim}")
        print(f"  Batch size: {datamodule.batch_size}")

        return True
    except Exception as e:
        print(f"✗ Creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_shd_adding_datamodule():
    """Test SHD-Adding variant."""
    print("\nTesting SHD-Adding DataModule...")
    try:
        from elmneuron.shd import SHDDataModule

        # Create SHD-Adding DataModule
        datamodule = SHDDataModule(
            data_dir="/tmp/test_shd_cache",
            dataset_variant="shd_adding",
            batch_size=128,
            batches_per_epoch=100,
        )

        print(f"✓ SHD-Adding DataModule created successfully")
        print(f"  Dataset variant: {datamodule.dataset_variant}")
        print(f"  Num classes: {datamodule.num_classes}")  # Should be 19 (0-18)
        print(f"  Batches per epoch: {datamodule.batches_per_epoch}")

        return True
    except Exception as e:
        print(f"✗ Creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_datamodule_properties():
    """Test DataModule property accessors."""
    print("\nTesting DataModule properties...")
    try:
        from elmneuron.neuronio import NeuronIODataModule
        from elmneuron.transforms import NeuronIORouting

        routing = NeuronIORouting(
            num_input=1278,
            num_branch=45,
            num_synapse_per_branch=100,
        )

        datamodule = NeuronIODataModule(
            train_files=["dummy.pkl"],
            routing=routing,
        )

        # Test properties
        assert datamodule.input_dim == 4500, "Incorrect input_dim"
        assert datamodule.num_classes == 2, "Incorrect num_classes"
        assert (
            datamodule.get_routing_info() is not None
        ), "Routing info should not be None"

        print("✓ All property accessors work correctly")
        return True
    except AssertionError as e:
        print(f"✗ Property test failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing ELM DataModules")
    print("=" * 60)

    results = []

    # NeuronIO tests
    results.append(("NeuronIO Import", test_neuronio_datamodule_import()))
    results.append(("NeuronIO Creation", test_neuronio_datamodule_creation()))
    results.append(("NeuronIO Routing", test_neuronio_routing()))

    # SHD tests
    results.append(("SHD Import", test_shd_datamodule_import()))
    results.append(("SHD Creation", test_shd_datamodule_creation()))
    results.append(("SHD-Adding", test_shd_adding_datamodule()))

    # Property tests
    results.append(("DataModule Properties", test_datamodule_properties()))

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
        sys.exit(1)
