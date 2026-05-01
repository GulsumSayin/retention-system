"""
llama_server_manager.py
llama.cpp server'ı otomatik başlatan ve durumunu kontrol eden yardımcı modül.

app.py tarafından import edilir; kullanıcının elle terminal komutu çalıştırmasına
gerek kalmaz.
"""

import os
import sys
import time
import subprocess
import requests
import logging

logger = logging.getLogger(__name__)

import json
import os

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json bulunmadı: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Config'i global olarak yükle
_config = load_config()
_LLAMA_EXE = _config['llama']['exe_path']
_MODEL_PATH = _config['llama']['model_path']
_PORT = _config['llama']['port']
_HEALTH_URL = f"http://localhost:{_PORT}/health"
_CONTEXT = _config['llama']['context']
_MAX_WAIT = _config['llama']['max_wait_seconds']

# Global process referansı — Streamlit her rerun'da yeniden başlatmamak için
_server_process = None


def is_server_running() -> bool:
    """llama.cpp server'ın ayakta olup olmadığını kontrol eder."""
    try:
        r = requests.get(_HEALTH_URL, timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def start_server() -> bool:
    """
    llama.cpp server'ı arka planda başlatır.
    Zaten çalışıyorsa dokunmaz.
    Başarılı ise True, başarısız ise False döner.
    """
    global _server_process

    if is_server_running():
        return True

    if not os.path.exists(_LLAMA_EXE):
        logger.error("llama-server.exe bulunamadı: %s", _LLAMA_EXE)
        return False

    if not os.path.exists(_MODEL_PATH):
        logger.error("Model dosyası bulunamadı: %s", _MODEL_PATH)
        return False

    try:
        popen_kwargs = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        _server_process = subprocess.Popen(
            [
                _LLAMA_EXE,
                "-m", _MODEL_PATH,
                "--port", str(_PORT),
                "-c", str(_CONTEXT),
                "--log-disable",
            ],
            **popen_kwargs,
        )
    except Exception as exc:
        logger.error("Server başlatılamadı: %s", exc)
        return False

    # Server ayağa kalkana kadar bekle
    for _ in range(_MAX_WAIT):
        time.sleep(1)
        if is_server_running():
            return True

    logger.error("Server %d saniye içinde yanıt vermedi.", _MAX_WAIT)
    return False


def stop_server() -> None:
    """Server process'ini sonlandırır (isteğe bağlı temizlik)."""
    global _server_process
    if _server_process and _server_process.poll() is None:
        _server_process.terminate()
        _server_process = None