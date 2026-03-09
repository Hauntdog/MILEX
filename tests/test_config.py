import json
import os
from pathlib import Path
from milex.config import load_config, save_config

def test_config_generation(tmp_path, monkeypatch):
    """Test that a config file is created with a daemon token if it doesn't exist."""
    # Mock CONFIG_DIR in the config module
    import milex.config
    monkeypatch.setattr(milex.config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(milex.config, "CONFIG_FILE", tmp_path / "config.json")
    
    cfg = load_config()
    
    assert "daemon_token" in cfg
    assert len(cfg["daemon_token"]) == 32 # hex string of 16 bytes
    assert (tmp_path / "config.json").exists()

def test_config_persistence(tmp_path, monkeypatch):
    """Test that config changes are saved correctly."""
    import milex.config
    monkeypatch.setattr(milex.config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(milex.config, "CONFIG_FILE", tmp_path / "config.json")
    
    cfg = load_config()
    cfg["theme"] = "cyberpunk"
    save_config(cfg)
    
    new_cfg = load_config()
    assert new_cfg["theme"] == "cyberpunk"
