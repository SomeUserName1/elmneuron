#!/usr/bin/env python
"""
Test script for LRA DataModules.
Verifies that LRA DataModules can be instantiated and provide correct metadata.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_lra_imports():
    """Test that LRA DataModules can be imported."""
    print("Testing LRA DataModule imports...")
    try:
        from elmneuron.lra import (
            ListOpsDataModule,
            LRAImageDataModule,
            LRAPathfinderDataModule,
            LRARetrievalDataModule,
            LRATextDataModule,
        )

        print("✓ All LRA DataModules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_listops_creation():
    """Test ListOps DataModule creation."""
    print("\nTesting ListOps DataModule creation...")
    try:
        from elmneuron.lra import ListOpsDataModule

        datamodule = ListOpsDataModule(
            data_dir="/tmp/test_lra",
            batch_size=32,
            download=False,  # Don't download in test
        )

        info = datamodule.get_dataset_info()
        print(f"✓ ListOps DataModule created successfully")
        print(f"  Task: {info['task_name']}")
        print(f"  Num classes: {info['num_classes']}")
        print(f"  Sequence length: {info['sequence_length']}")
        print(f"  Vocab size: {info['vocab_size']}")

        assert info["num_classes"] == 10, "ListOps should have 10 classes"
        assert info["sequence_length"] == 2048, "ListOps sequences should be 2048"

        return True
    except Exception as e:
        print(f"✗ Creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_text_creation():
    """Test LRA Text DataModule creation."""
    print("\nTesting LRA Text DataModule creation...")
    try:
        from elmneuron.lra import LRATextDataModule

        datamodule = LRATextDataModule(
            data_dir="/tmp/test_lra",
            batch_size=32,
            download=False,
        )

        info = datamodule.get_dataset_info()
        print(f"✓ LRA Text DataModule created successfully")
        print(f"  Task: {info['task_name']}")
        print(f"  Num classes: {info['num_classes']}")
        print(f"  Sequence length: {info['sequence_length']}")
        print(f"  Vocab size: {info['vocab_size']}")

        assert info["num_classes"] == 2, "Text should have 2 classes (sentiment)"
        assert info["sequence_length"] == 4096, "Text sequences should be 4096"
        assert info["vocab_size"] == 256, "Text should be byte-level (256)"

        return True
    except Exception as e:
        print(f"✗ Creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_retrieval_creation():
    """Test LRA Retrieval DataModule creation."""
    print("\nTesting LRA Retrieval DataModule creation...")
    try:
        from elmneuron.lra import LRARetrievalDataModule

        datamodule = LRARetrievalDataModule(
            data_dir="/tmp/test_lra",
            batch_size=16,
            download=False,
        )

        info = datamodule.get_dataset_info()
        print(f"✓ LRA Retrieval DataModule created successfully")
        print(f"  Task: {info['task_name']}")
        print(f"  Num classes: {info['num_classes']}")
        print(f"  Sequence length: {info['sequence_length']}")
        print(f"  Vocab size: {info['vocab_size']}")

        assert (
            info["num_classes"] == 2
        ), "Retrieval should have 2 classes (match/no-match)"
        assert info["sequence_length"] == 8192, "Retrieval sequences should be 8192"

        return True
    except Exception as e:
        print(f"✗ Creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_image_creation():
    """Test LRA Image DataModule creation."""
    print("\nTesting LRA Image DataModule creation...")
    try:
        from elmneuron.lra import LRAImageDataModule

        datamodule = LRAImageDataModule(
            data_dir="/tmp/test_lra",
            batch_size=64,
            download=False,
        )

        info = datamodule.get_dataset_info()
        print(f"✓ LRA Image DataModule created successfully")
        print(f"  Task: {info['task_name']}")
        print(f"  Num classes: {info['num_classes']}")
        print(f"  Sequence length: {info['sequence_length']}")
        print(f"  Vocab size: {info['vocab_size']}")

        assert info["num_classes"] == 10, "Image should have 10 classes (CIFAR-10)"
        assert info["sequence_length"] == 1024, "Image sequences should be 1024 (32×32)"

        return True
    except Exception as e:
        print(f"✗ Creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_pathfinder_creation():
    """Test LRA Pathfinder DataModule creation."""
    print("\nTesting LRA Pathfinder DataModule creation...")
    try:
        from elmneuron.lra import LRAPathfinderDataModule

        for difficulty in ["easy", "medium", "hard"]:
            datamodule = LRAPathfinderDataModule(
                data_dir="/tmp/test_lra",
                difficulty=difficulty,
                batch_size=64,
                download=False,
            )

            info = datamodule.get_dataset_info()
            print(f"✓ Pathfinder ({difficulty}) DataModule created")
            print(f"    Sequence length: {info['sequence_length']}")

            assert info["num_classes"] == 2, "Pathfinder should have 2 classes"
            assert (
                info["sequence_length"] == 1024
            ), "Pathfinder sequences should be 1024"

        print("✓ All Pathfinder difficulties work")
        return True
    except Exception as e:
        print(f"✗ Creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sequence_lengths():
    """Test that all tasks have correct sequence lengths."""
    print("\nTesting LRA sequence lengths...")
    try:
        from elmneuron.lra import (
            ListOpsDataModule,
            LRAImageDataModule,
            LRAPathfinderDataModule,
            LRARetrievalDataModule,
            LRATextDataModule,
        )

        expected_lengths = {
            "ListOps": 2048,
            "Text": 4096,
            "Retrieval": 8192,
            "Image": 1024,
            "Pathfinder": 1024,
        }

        datamodules = {
            "ListOps": ListOpsDataModule(data_dir="/tmp/test_lra", download=False),
            "Text": LRATextDataModule(data_dir="/tmp/test_lra", download=False),
            "Retrieval": LRARetrievalDataModule(
                data_dir="/tmp/test_lra", download=False
            ),
            "Image": LRAImageDataModule(data_dir="/tmp/test_lra", download=False),
            "Pathfinder": LRAPathfinderDataModule(
                data_dir="/tmp/test_lra", download=False
            ),
        }

        print("✓ Sequence length verification:")
        for name, dm in datamodules.items():
            info = dm.get_dataset_info()
            expected = expected_lengths[name]
            actual = info["sequence_length"]
            status = "✓" if actual == expected else "✗"
            print(f"  {status} {name}: {actual} (expected: {expected})")

            assert actual == expected, f"{name} has wrong sequence length"

        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_download_function():
    """Test that download function is available."""
    print("\nTesting LRA download function...")
    try:
        from elmneuron.lra.lra_datamodule import download_lra_dataset

        print("✓ Download function available")
        print("  Note: download_lra_dataset() will download ~7GB when called")
        print("  Usage: download_lra_dataset(Path('~/.cache/elmneuron/lra'))")

        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing LRA DataModules")
    print("=" * 60)

    results = []

    # Basic tests
    results.append(("LRA Imports", test_lra_imports()))

    # Only run other tests if imports succeeded
    if results[0][1]:
        results.append(("ListOps Creation", test_listops_creation()))
        results.append(("Text Creation", test_text_creation()))
        results.append(("Retrieval Creation", test_retrieval_creation()))
        results.append(("Image Creation", test_image_creation()))
        results.append(("Pathfinder Creation", test_pathfinder_creation()))
        results.append(("Sequence Lengths", test_sequence_lengths()))
        results.append(("Download Function", test_download_function()))
    else:
        print("\n⚠ Skipping remaining tests due to import failure")

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
        print("\nNote: To use LRA datasets, you need to download them first:")
        print("  1. Create a DataModule with download=True")
        print("  2. Call datamodule.prepare_data()")
        print("  3. This will download ~7GB from Google Cloud Storage")
        sys.exit(0)
    else:
        print(f"✗ {total - passed} test(s) failed")
        sys.exit(1)
