import pytest
from pathlib import Path
from milex.tools import ToolExecutor

def test_validate_path(tmp_path):
    """Test path validation logic."""
    from milex.tools import _validate_path
    
    config = {"allowed_root": str(tmp_path)}
    safe_path = tmp_path / "test.txt"
    safe_path.touch()
    
    # Within root
    assert _validate_path(str(safe_path), config) == safe_path.resolve()
    
    # Outside root
    unsafe_path = Path("/etc/passwd")
    with pytest.raises(PermissionError):
        _validate_path(str(unsafe_path), config)

def test_tool_execute_read_write(tmp_path):
    """Test basic file tools."""
    config = {"allowed_root": str(tmp_path)}
    executor = ToolExecutor(config)
    
    test_file = tmp_path / "hello.txt"
    content = "Hello, MILEX!"
    
    # Write
    write_res = executor.execute("write_file", {"path": str(test_file), "content": content})
    assert "success" in write_res
    assert test_file.read_text() == content
    
    # Read
    read_res = executor.execute("read_file", {"path": str(test_file)})
    assert read_res["content"] == content
