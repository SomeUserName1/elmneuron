#!/usr/bin/env python
"""
Test script for visualization callbacks.
Verifies that callbacks can be imported and instantiated.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_callback_imports():
    """Test that callbacks can be imported."""
    print("Testing callback imports...")
    try:
        from elmneuron.callbacks import (
            MemoryDynamicsCallback,
            SequenceVisualizationCallback,
            StateRecorderCallback,
        )

        print("✓ All callbacks imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_state_recorder_creation():
    """Test StateRecorderCallback creation."""
    print("\nTesting StateRecorderCallback creation...")
    try:
        from elmneuron.callbacks import StateRecorderCallback

        callback = StateRecorderCallback(
            record_every_n_epochs=5,
            num_samples=8,
            record_train=True,
            record_val=True,
        )

        print("✓ StateRecorderCallback created successfully")
        print(f"  Record every {callback.record_every_n_epochs} epochs")
        print(f"  Number of samples: {callback.num_samples}")
        print(f"  Record train: {callback.record_train}")
        print(f"  Record val: {callback.record_val}")

        assert callback.record_every_n_epochs == 5
        assert callback.num_samples == 8

        return True
    except Exception as e:
        print(f"✗ Creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sequence_visualization_creation():
    """Test SequenceVisualizationCallback creation."""
    print("\nTesting SequenceVisualizationCallback creation...")
    try:
        from elmneuron.callbacks import SequenceVisualizationCallback

        # Test different task types
        task_types = ["classification", "regression", "timeseries"]

        for task_type in task_types:
            callback = SequenceVisualizationCallback(
                log_every_n_epochs=10,
                num_samples=4,
                task_type=task_type,
            )

            print(f"✓ SequenceVisualizationCallback created for '{task_type}'")
            assert callback.task_type == task_type

        print("✓ All task types work correctly")
        return True
    except Exception as e:
        print(f"✗ Creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_memory_dynamics_creation():
    """Test MemoryDynamicsCallback creation."""
    print("\nTesting MemoryDynamicsCallback creation...")
    try:
        from elmneuron.callbacks import MemoryDynamicsCallback

        callback = MemoryDynamicsCallback(
            log_every_n_epochs=10,
            num_samples=2,
            log_to_wandb=False,
        )

        print("✓ MemoryDynamicsCallback created successfully")
        print(f"  Log every {callback.log_every_n_epochs} epochs")
        print(f"  Number of samples: {callback.num_samples}")

        assert callback.log_every_n_epochs == 10
        assert callback.num_samples == 2

        return True
    except Exception as e:
        print(f"✗ Creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_callback_with_save_dir():
    """Test callbacks with save directory."""
    print("\nTesting callbacks with save directory...")
    try:
        import tempfile

        from elmneuron.callbacks import (
            MemoryDynamicsCallback,
            SequenceVisualizationCallback,
            StateRecorderCallback,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test StateRecorderCallback
            callback1 = StateRecorderCallback(save_dir=tmpdir)
            assert callback1.save_dir is not None
            print("✓ StateRecorderCallback with save_dir works")

            # Test SequenceVisualizationCallback
            callback2 = SequenceVisualizationCallback(save_dir=tmpdir)
            assert callback2.save_dir is not None
            print("✓ SequenceVisualizationCallback with save_dir works")

            # Test MemoryDynamicsCallback
            callback3 = MemoryDynamicsCallback(save_dir=tmpdir)
            assert callback3.save_dir is not None
            print("✓ MemoryDynamicsCallback with save_dir works")

        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_callback_inheritance():
    """Test that callbacks inherit from Lightning Callback."""
    print("\nTesting callback inheritance...")
    try:
        from pytorch_lightning.callbacks import Callback

        from elmneuron.callbacks import (
            MemoryDynamicsCallback,
            SequenceVisualizationCallback,
            StateRecorderCallback,
        )

        callback1 = StateRecorderCallback()
        callback2 = SequenceVisualizationCallback()
        callback3 = MemoryDynamicsCallback()

        assert isinstance(callback1, Callback)
        assert isinstance(callback2, Callback)
        assert isinstance(callback3, Callback)

        print("✓ All callbacks inherit from Lightning Callback")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_callback_methods():
    """Test that callbacks have required Lightning methods."""
    print("\nTesting callback methods...")
    try:
        from elmneuron.callbacks import (
            MemoryDynamicsCallback,
            SequenceVisualizationCallback,
            StateRecorderCallback,
        )

        # Check StateRecorderCallback methods
        callback1 = StateRecorderCallback()
        assert hasattr(callback1, "on_train_batch_end")
        assert hasattr(callback1, "on_validation_batch_end")
        assert hasattr(callback1, "on_train_epoch_end")
        print("✓ StateRecorderCallback has required methods")

        # Check SequenceVisualizationCallback methods
        callback2 = SequenceVisualizationCallback()
        assert hasattr(callback2, "on_validation_epoch_end")
        print("✓ SequenceVisualizationCallback has required methods")

        # Check MemoryDynamicsCallback methods
        callback3 = MemoryDynamicsCallback()
        assert hasattr(callback3, "on_validation_epoch_end")
        print("✓ MemoryDynamicsCallback has required methods")

        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Visualization Callbacks")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Callback Imports", test_callback_imports()))

    # Only run other tests if imports succeeded
    if results[0][1]:
        results.append(
            ("StateRecorderCallback Creation", test_state_recorder_creation())
        )
        results.append(
            (
                "SequenceVisualizationCallback Creation",
                test_sequence_visualization_creation(),
            )
        )
        results.append(
            ("MemoryDynamicsCallback Creation", test_memory_dynamics_creation())
        )
        results.append(("Callbacks with Save Dir", test_callback_with_save_dir()))
        results.append(("Callback Inheritance", test_callback_inheritance()))
        results.append(("Callback Methods", test_callback_methods()))
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
        print("\nCallbacks are ready to use with PyTorch Lightning:")
        print("  from elmneuron.callbacks import StateRecorderCallback")
        print("  callback = StateRecorderCallback()")
        print("  trainer = pl.Trainer(callbacks=[callback])")
        sys.exit(0)
    else:
        print(f"✗ {total - passed} test(s) failed")
        sys.exit(1)
