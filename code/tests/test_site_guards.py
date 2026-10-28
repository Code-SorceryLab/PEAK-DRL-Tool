"""Guards around the published site and the local servers: the Reproduce tab names a commit a reader
can check out, publish only ships committed results, a failed build does not block the next one, and
other websites cannot start jobs on this machine."""
import subprocess

import pytest

from code.neuro import command_center
from code.stats import telemetry


def git(cwd, *args) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repo with one committed result and a built site; its commit is on no remote yet."""
    for who in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{who}_NAME", "T")
        monkeypatch.setenv(f"GIT_{who}_EMAIL", "t@example.com")
    r = tmp_path / "repo"
    (r / "runs" / "balance").mkdir(parents=True)
    (r / "runs" / "balance" / "report_x.json").write_text("{}")
    (r / "site").mkdir()
    (r / "site" / "index.html").write_text("new")
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "add", "runs")
    git(r, "commit", "-q", "-m", "results")
    return r


def test_the_reproduce_tab_checks_out_the_commit_it_was_built_from():
    sha = command_center._git(".", "rev-parse", "HEAD", check=False)
    if not sha:
        pytest.skip("not a git checkout (a cluster copy unpacked from a tar)")
    html = command_center.reproduce_page("mario", [])
    assert f"cd PEAK-DRL-Tool\ngit checkout {sha}\n" in html     # a clone alone is main


def test_a_copy_outside_git_still_builds_the_reproduce_tab(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                                  # no .git: there is no commit to name
    assert "git checkout" not in command_center.reproduce_page("mario", [])


def test_publish_names_its_source_commit_and_refuses_one_no_remote_has(repo, tmp_path):
    with pytest.raises(SystemExit, match="remote"):
        command_center.publish(str(repo / "site"), repo=str(repo))
    clone = tmp_path / "clone"
    git(tmp_path, "clone", "-q", str(repo), str(clone))                    # origin/main now holds the commit
    commit = command_center.publish(str(repo / "site"), repo=str(clone))
    assert git(repo, "rev-parse", "HEAD") in git(clone, "log", "-1", "--format=%s", commit)


def test_publish_refuses_an_uncommitted_run_the_how_it_works_tab_lists(repo):
    (repo / "runs" / "mario_Mario1-1").mkdir()
    (repo / "runs" / "mario_Mario1-1" / "state.json").write_text("{}")      # a scratch run, never committed
    with pytest.raises(SystemExit, match="uncommitted"):
        command_center.publish(str(repo / "site"), repo=str(repo))


def test_a_build_that_failed_part_way_does_not_block_the_next(tmp_path, monkeypatch):
    out = tmp_path / "site"

    def fail():
        (out / "half.html").write_text("")                                 # something written, then it breaks
        raise RuntimeError("build failed")
    monkeypatch.setattr(command_center.report, "load_data", fail)
    for _ in range(2):                                                     # the second build replaces the first
        with pytest.raises(RuntimeError):
            command_center.build_site(str(out), str(tmp_path / "videos"))


def test_only_pages_on_this_machine_reach_the_local_servers():
    page = {"Host": "127.0.0.1:8003", "Sec-Fetch-Site": "same-origin"}
    assert telemetry.local_request(page, 8003, changes_state=True)
    assert telemetry.local_request({"Host": "localhost:8003", "Sec-Fetch-Site": "none"}, 8003, changes_state=True)  # typed
    other_port = {"Host": "127.0.0.1:8003", "Sec-Fetch-Site": "same-site"}   # a page on another local port
    assert not telemetry.local_request(other_port, 8003, changes_state=True)
    assert not telemetry.local_request({"Host": "127.0.0.1:8003"}, 8003, changes_state=True)   # no header: fail closed
    assert not telemetry.local_request({"Host": "evil.example:8003"}, 8003, changes_state=False)   # DNS rebinding
    cross = {"Host": "127.0.0.1:8003", "Sec-Fetch-Site": "cross-site"}
    assert not telemetry.local_request(cross, 8003, changes_state=True)   # another site's link to /watch
    assert telemetry.local_request(cross, 8003, changes_state=False)      # another site's link to a page
