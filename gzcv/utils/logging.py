import logging
import os
from datetime import datetime

import coloredlogs
import git
import sh


def setup_logger(log_dir, file_name="main.log", exist_ok=True, level="info"):
    """
    [Summary]: logging all to one directory, such as git info, pth file, and results.
    [Args]:
        log_dir          : relative path to your desired logging directory.
        forbit_overwrite : if false, existing directory might overwrite.
        record_git       : if true, record git SHA and git diff.
    """
    os.makedirs(log_dir, exist_ok=exist_ok)

    logger = logging.getLogger()
    stream_fmt = "[%(levelname)s] - %(message)s"
    coloredlogs.install(level=level.upper(), logger=logger, fmt=stream_fmt)

    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name) 15s - %(message)s"
    )
    log_path = os.path.join(log_dir, file_name)
    file_hdlr = logging.FileHandler(log_path)
    file_hdlr.setFormatter(file_formatter)
    file_hdlr.setLevel(logging.INFO)
    logger.addHandler(file_hdlr)


def record_gitinfo(log_dir) -> None:
    """
    [Summary]: logging git SHA, commit date.
               The result of "git diff" will saved to compareHead.diff.
    """
    try:
        repo = git.Repo(search_parent_directories=True)
        git_date = datetime.fromtimestamp(repo.head.object.committed_date).strftime(
            "%Y-%m-%d"
        )
        git_sha = repo.head.object.hexsha[:8]
        git_message = repo.head.object.message.strip()
        msg = f"Source is from Commit {git_sha} ({git_date}): {git_message}"

        if log_dir is not None:
            sh.git.diff(_out=os.path.join(log_dir, "compareHead.diff"))

    except git.exc.InvalidGitRepositoryError:
        msg = "Failed to record git status"

    return msg


def get_log_dir(logger):
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            basefile = handler.baseFilename
            log_dir = os.path.dirname(basefile)
            return log_dir
    return None
