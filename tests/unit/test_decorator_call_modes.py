from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from pydantic import BaseModel
import stryx

# --- Setup ---

class MyConfig(BaseModel):
    val: int = 1

@pytest.fixture
def decorated_func():
    """Returns a fresh decorated function and its underlying mock for each test."""
    mock_impl = MagicMock(return_value="programmatic_result")
    
    @stryx.cli(schema=MyConfig)
    def my_main(cfg: MyConfig, **kwargs):
        return mock_impl(cfg, **kwargs)
        
    return my_main, mock_impl

# --- Tests for Programmatic Usage (Direct Call) ---

def test_direct_call_success(decorated_func):
    """Test calling the decorated function with a valid config object."""
    main, impl = decorated_func
    cfg = MyConfig(val=10)
    
    # 1. Call directly
    result = main(cfg)
    
    # 2. Verify behavior
    assert result == "programmatic_result"
    impl.assert_called_once_with(cfg)

def test_direct_call_kwargs(decorated_func):
    """Test calling with kwargs (common in some frameworks)."""
    main, impl = decorated_func
    cfg = MyConfig(val=20)
    
    # 1. Call with kwargs
    result = main(cfg=cfg, extra="foo")
    
    # 2. Verify behavior
    assert result == "programmatic_result"
    impl.assert_called_once_with(cfg, extra="foo")

def test_direct_call_invalid_type(decorated_func):
    """Test calling with an invalid type (should not trigger CLI)."""
    @stryx.cli(schema=MyConfig)
    def fragile_main(cfg: MyConfig):
        return cfg.val

    with pytest.raises(AttributeError):
        fragile_main(123)

# --- Tests for CLI Usage (No Arguments) ---

@patch("stryx.cli.dispatch")
@patch("sys.argv", ["script.py", "run", "my_recipe"])
def test_cli_invocation(mock_dispatch, decorated_func):
    """
    Test calling with NO arguments.
    
    This should:
    1. Detect no args.
    2. Call stryx.cli.dispatch with context components.
    """
    main, impl = decorated_func
    mock_dispatch.return_value = "cli_result"
    
    # 1. Call as CLI
    result = main()
    
    # 2. Verify behavior
    assert result == "cli_result"
    
    # The underlying function should NOT be called yet (dispatch handles that)
    impl.assert_not_called()
    
    # Verify dispatch was called with correct arguments
    mock_dispatch.assert_called_once()
    kwargs = mock_dispatch.call_args.kwargs
    
    assert kwargs["schema"] == MyConfig
    assert kwargs["func"].__name__ == "my_main"
    assert kwargs["configs_dir"].name == "configs"
    assert kwargs["runs_dir"].name == "runs"
    assert kwargs["argv"] == ["run", "my_recipe"]

@patch("stryx.cli.dispatch")
def test_cli_invocation_default_args(mock_dispatch, decorated_func):
    """Test that CLI invocation passes the correct default paths."""
    main, _ = decorated_func
    main()
    
    kwargs = mock_dispatch.call_args.kwargs
    assert kwargs["configs_dir"].name == "configs"
    assert kwargs["runs_dir"].name == "runs"