"""
storage package
---------------
Database persistence for runs, recruiter feedback, and candidate evaluation memory.
"""

from storage.db import (
    get_all_feedback,
    get_feedback_for_role,
    get_run,
    init_db,
    list_runs,
    record_feedback,
    save_run,
)

__all__ = [
    "init_db",
    "save_run",
    "get_run",
    "list_runs",
    "record_feedback",
    "get_feedback_for_role",
    "get_all_feedback",
]
