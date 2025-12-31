from __future__ import annotations
import sys
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
    """
    Test calling with an invalid type.
    
    Because we removed the `isinstance` check, this should NOT trigger the CLI.
    It should call the underlying function directly, which will likely raise
    a standard Python error (like AttributeError) if it tries to use the object.
    """
    # Define a function that actually uses the input to trigger an error
    @stryx.cli(schema=MyConfig)
    def fragile_main(cfg: MyConfig):
        return cfg.val  # Will fail if cfg is not a Config

    # Pass an integer instead of Config
    # Expect: AttributeError (int has no attribute 'val'), NOT SystemExit or CLI parsing error
    with pytest.raises(AttributeError):
        fragile_main(123)

# --- Tests for CLI Usage (No Arguments) ---

@patch("stryx.decorator.dispatch")
@patch("sys.argv", ["script.py", "run", "my_recipe"])
def test_cli_invocation(mock_dispatch, decorated_func):
    """
    Test calling with NO arguments.
    
    This should:
    1. Detect no args.
    2. Create a Ctx (Context).
    3. Call stryx.decorator.dispatch.
    """
    main, impl = decorated_func
    mock_dispatch.return_value = "cli_result"
    
    # 1. Call as CLI
    result = main()
    
    # 2. Verify behavior
    assert result == "cli_result"
    
    # The underlying function should NOT be called yet (dispatch handles that)
    impl.assert_not_called()
    
    # Verify dispatch was called with correct context
    mock_dispatch.assert_called_once()
    ctx_arg = mock_dispatch.call_args[0][0]
    assert ctx_arg.schema == MyConfig
    assert ctx_arg.func.__name__ == "my_main"
    
    # Verify sys.argv was passed (excluding script name)
    args_passed = mock_dispatch.call_args[0][1]
    assert args_passed == ["run", "my_recipe"]

@patch("stryx.decorator.dispatch")
def test_cli_invocation_default_args(mock_dispatch, decorated_func):
    """Test that CLI invocation passes the correct default paths."""
    main, _ = decorated_func
    
    main()
    
    ctx = mock_dispatch.call_args[0][0]
    assert ctx.configs_dir.name == "configs"
    assert ctx.runs_dir.name == "runs"
