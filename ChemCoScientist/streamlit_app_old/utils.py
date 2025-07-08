import os
import shutil
import tempfile
import threading
import time

from dotenv import load_dotenv

from definitions import ROOT_DIR, CONFIG_PATH

load_dotenv(CONFIG_PATH)

PATH_TO_TEMP_FILES = os.environ["PATH_TO_TEMP_FILES"]
INACTIVITY_WINDOW_SECONDS = 6 * 60 * 60  # 6 hours


def update_activity(session_folder: str) -> None:
    with open(os.path.join(session_folder, ".last_activity"), "w") as f:
        f.write(str(time.time()))


def cleanup_expired_sessions(base_dir: str = None) -> None:
    """
    Deletes session folders that have been inactive for more than the inactivity window.
    """
    if base_dir is None:
        base_dir = os.path.join(ROOT_DIR, PATH_TO_TEMP_FILES)
    if not os.path.exists(base_dir):
        return
    now = time.time()
    for session_id in os.listdir(base_dir):
        session_folder = os.path.join(base_dir, session_id)
        last_activity_file = os.path.join(session_folder, ".last_activity")
        if os.path.isdir(session_folder) and os.path.exists(last_activity_file):
            with open(last_activity_file, "r") as f:
                last_activity = float(f.read().strip())
            if now - last_activity > INACTIVITY_WINDOW_SECONDS:
                shutil.rmtree(session_folder)


def start_cleanup_thread(interval: int = 60 * 60) -> threading.Thread:
    """
    Starts a background thread that runs cleanup_expired_sessions every `interval` seconds.
    """
    def cleanup_loop():
        while True:
            cleanup_expired_sessions()
            time.sleep(interval)
    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()
    return thread

