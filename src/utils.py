import logging
import os
import sqlite3

def check_paths(paths) -> None:
    for path in paths.values():
        if(not os.path.exists(path)):
            os.makedirs(path)

def setup_loggers(log_path) -> None:
    '''
    user logger
    '''
    user_logger = logging.getLogger("user_requests")
    user_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(log_path, "user_requests.log"), encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | user_id=%(user_id)s | username=%(username)s | %(message)s"))

    user_logger.addHandler(file_handler)
    user_logger.propagate = False
    '''
    system logger
    '''
    system_logger = logging.getLogger("system")
    system_logger.setLevel(logging.INFO)

    system_file_handler = logging.FileHandler(os.path.join(log_path, "system.log"), encoding="utf-8")
    system_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

    system_logger.addHandler(system_file_handler)
    system_logger.propagate = False

def init_db(db_path):
    conn = sqlite3.connect(os.path.join(db_path, "user_votes.db"))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS votes (
        user_id INTEGER,
        joke_id TEXT,
        action TEXT,
        reacted_at TEXT,
        PRIMARY KEY (user_id, joke_id)
    );
    """)
    conn.commit()
    return conn