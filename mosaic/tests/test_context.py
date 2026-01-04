"""Tests for MosaicContext."""

import pytest
from mosaic.context import MosaicContext


class TestMosaicContext:
    """Test MosaicContext singleton."""
    
    def setup_method(self):
        """Reset context before each test."""
        MosaicContext.reset()
    
    def test_not_initialized_error(self):
        """Should raise error if get() called before init()."""
        with pytest.raises(RuntimeError, match="not initialized"):
            MosaicContext.get()
    
    def test_double_init_error(self):
        """Should raise error if init() called twice."""
        # Skip if distributed not available
        pytest.skip("Requires distributed environment")


class TestAxisSpec:
    """Test AxisSpec validation."""
    
    def test_valid_backends(self):
        from mosaic.spec import AxisSpec
        
        spec = AxisSpec(axis=1, backend="local")
        assert spec.backend == "local"
        
        spec = AxisSpec(axis=1, backend="ring")
        assert spec.backend == "ring"
    
    def test_invalid_backend(self):
        from mosaic.spec import AxisSpec
        
        with pytest.raises(ValueError, match="Unknown backend"):
            AxisSpec(axis=1, backend="invalid")
    
    def test_magi_not_implemented(self):
        from mosaic.spec import AxisSpec
        
        with pytest.raises(NotImplementedError):
            AxisSpec(axis=1, backend="magi")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

