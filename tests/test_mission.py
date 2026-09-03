"""A mission scopes work. The two things that must not slip are its identity
and its scroll selection.

The selection freezes on the first run, not at creation, so most of these come
in pairs: the draft case, where an edit is free, and the frozen case, where the
same edit is an amendment that has to say why.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framework.contracts import mission


def give_it_a_run(directory: Path, name: str = "run-0001") -> None:
    """The one thing that freezes a selection: a receipt under the mission."""
    run = directory / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "INK_RECEIPT.json").write_text("{}")


def test_a_mission_is_a_directory_with_a_manifest(tmp_path: Path):
    manifest = mission.create(tmp_path, mission_id="first-letters-826", name="PHerc0826",
                              scrolls=["PHerc0826"])
    assert (tmp_path / "first-letters-826" / "MISSION.json").is_file()
    assert manifest["scrolls"] == ["PHerc0826"]
    # Nothing has run, so the selection is still a draft.
    assert manifest["scrolls_frozen_at_utc"] is None
    assert mission.load(tmp_path / "first-letters-826")["name"] == "PHerc0826"


def test_ids_stay_boring(tmp_path: Path):
    """The id becomes a directory name and part of a job id."""
    for bad in ("Has Capitals", "-leading", "trailing-", "a", "with/slash", "with spaces"):
        with pytest.raises(mission.MissionError):
            mission.create(tmp_path, mission_id=bad, name="x", scrolls=["PHerc0001"])


def test_a_mission_needs_a_name_but_not_yet_a_scroll(tmp_path: Path):
    """Choosing scrolls is P0's job; a mission that has not chosen is a real
    state, and its freeze timestamp stays empty until something is selected."""
    empty = mission.create(tmp_path, mission_id="empty-one", name="x", scrolls=[])
    assert empty["scrolls"] == []
    assert empty["scrolls_frozen_at_utc"] is None
    with pytest.raises(mission.MissionError):
        mission.create(tmp_path, mission_id="nameless", name="  ", scrolls=["PHerc0001"])


def test_duplicate_scrolls_are_refused(tmp_path: Path):
    with pytest.raises(mission.MissionError, match="duplicate"):
        mission.create(tmp_path, mission_id="dupes", name="x",
                       scrolls=["PHerc0001", "PHerc0001"])


def test_creating_twice_refuses_rather_than_overwriting(tmp_path: Path):
    mission.create(tmp_path, mission_id="once", name="x", scrolls=["PHerc0001"])
    with pytest.raises(mission.MissionError, match="already exists"):
        mission.create(tmp_path, mission_id="once", name="y", scrolls=["PHerc0002"])


def test_widening_the_selection_records_what_and_why(tmp_path: Path):
    """A selection that grows silently makes every earlier negative unreadable."""
    mission.create(tmp_path, mission_id="grow", name="x", scrolls=["PHerc0001"])
    directory = tmp_path / "grow"
    give_it_a_run(directory)
    with pytest.raises(mission.MissionError, match="reason"):
        mission.amend_scrolls(directory, add=["PHerc0002"], reason="")
    give_it_a_run(directory)
    amended = mission.amend_scrolls(directory, add=["PHerc0002"],
                                    reason="the 2.4 um rescan landed")
    assert amended["scrolls"] == ["PHerc0001", "PHerc0002"]
    assert amended["amendments"][0]["added"] == ["PHerc0002"]
    assert "rescan" in amended["amendments"][0]["reason"]
    assert amended["scrolls_frozen_at_utc"]
    with pytest.raises(mission.MissionError, match="already in the selection"):
        mission.amend_scrolls(directory, add=["PHerc0001"], reason="again")


def test_discovery_reports_loose_runs_without_calling_them_a_mission(tmp_path: Path):
    """An existing installation keeps working; its runs are shown, not hidden,
    and not dressed up as something they are not."""
    loose = tmp_path / "pherc0139-old"
    loose.mkdir()
    (loose / "INK_RECEIPT.json").write_text("{}")
    mission.create(tmp_path, mission_id="real-one", name="x", scrolls=["PHerc0826"])

    found = {m["mission_id"]: m for m in mission.discover(tmp_path)}
    assert set(found) == {"real-one", "unfiled"}
    assert found["unfiled"]["implicit"] is True
    assert found["unfiled"]["scrolls_frozen_at_utc"] is None
    assert found["real-one"].get("implicit") is None


def test_state_transitions_are_bounded(tmp_path: Path):
    mission.create(tmp_path, mission_id="states", name="x", scrolls=["PHerc0001"])
    directory = tmp_path / "states"
    assert mission.set_state(directory, "archived")["state"] == "archived"
    with pytest.raises(mission.MissionError):
        mission.set_state(directory, "deleted")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_narrowing_records_what_left_and_why(tmp_path: Path):
    mission.create(tmp_path, mission_id="shrink", name="x",
                   scrolls=["PHerc0001", "PHerc0002", "PHerc0003"])
    directory = tmp_path / "shrink"
    give_it_a_run(directory)
    with pytest.raises(mission.MissionError, match="reason"):
        mission.remove_scrolls(directory, remove=["PHerc0002"], reason="")
    narrowed = mission.remove_scrolls(directory, remove=["PHerc0002"],
                                      reason="no public segment at this scale")
    assert narrowed["scrolls"] == ["PHerc0001", "PHerc0003"]
    assert narrowed["amendments"][0]["removed"] == ["PHerc0002"]


def test_a_scroll_that_produced_work_cannot_be_dropped(tmp_path: Path):
    """Otherwise the manifest disowns work whose receipts name this mission."""
    mission.create(tmp_path, mission_id="hasruns", name="x",
                   scrolls=["PHerc0001", "PHerc0002"])
    directory = tmp_path / "hasruns"
    with pytest.raises(mission.MissionError, match="cannot be removed"):
        mission.remove_scrolls(directory, remove=["PHerc0001"], reason="changed my mind",
                               protected={"PHerc0001"})
    assert mission.load(directory)["scrolls"] == ["PHerc0001", "PHerc0002"]


def test_a_mission_can_be_emptied_again(tmp_path: Path):
    """Empty is a valid state on the way in, so it is valid on the way out."""
    mission.create(tmp_path, mission_id="lastone", name="x", scrolls=["PHerc0001"])
    give_it_a_run(tmp_path / "lastone")
    emptied = mission.remove_scrolls(tmp_path / "lastone", remove=["PHerc0001"],
                                     reason="starting the selection over")
    assert emptied["scrolls"] == []
    assert emptied["amendments"][-1]["removed"] == ["PHerc0001"]


def test_a_draft_selection_is_edited_without_ceremony(tmp_path: Path):
    """No receipts means no claim, so there is nothing for a reason to protect.

    This is the common case -- picking scrolls for a mission that has not run
    yet -- and it used to demand a written justification for every checkbox.
    """
    mission.create(tmp_path, mission_id="draft", name="x", scrolls=["PHerc0001"])
    directory = tmp_path / "draft"

    added = mission.amend_scrolls(directory, add=["PHerc0002"])
    assert added["scrolls"] == ["PHerc0001", "PHerc0002"]
    removed = mission.remove_scrolls(directory, remove=["PHerc0001"])
    assert removed["scrolls"] == ["PHerc0002"]

    # A draft leaves no paper trail: there is no history of a decision nobody
    # could have relied on.
    assert removed["amendments"] == []
    assert removed["scrolls_frozen_at_utc"] is None
    assert mission.has_work(directory) is False


def test_the_first_run_is_what_freezes_the_selection(tmp_path: Path):
    """The same edit, before and after a receipt exists."""
    mission.create(tmp_path, mission_id="turns", name="x", scrolls=["PHerc0001"])
    directory = tmp_path / "turns"

    mission.amend_scrolls(directory, add=["PHerc0002"])
    assert mission.has_work(directory) is False

    give_it_a_run(directory)
    assert mission.has_work(directory) is True
    with pytest.raises(mission.MissionError, match="frozen"):
        mission.amend_scrolls(directory, add=["PHerc0003"])
    after = mission.amend_scrolls(directory, add=["PHerc0003"], reason="widening the sweep")
    assert after["amendments"][-1]["added"] == ["PHerc0003"]


def test_a_directory_without_receipts_does_not_count_as_work(tmp_path: Path):
    """An empty output directory is not a run; only a receipt is."""
    mission.create(tmp_path, mission_id="hollow", name="x", scrolls=["PHerc0001"])
    directory = tmp_path / "hollow"
    (directory / "started-but-produced-nothing").mkdir()
    assert mission.has_work(directory) is False
    mission.amend_scrolls(directory, add=["PHerc0002"])


def test_discovery_reports_whether_the_selection_is_frozen(tmp_path: Path):
    mission.create(tmp_path, mission_id="cold", name="x", scrolls=["PHerc0001"])
    mission.create(tmp_path, mission_id="warm", name="x", scrolls=["PHerc0001"])
    give_it_a_run(tmp_path / "warm")

    found = {m["mission_id"]: m for m in mission.discover(tmp_path)}
    assert found["cold"]["selection_frozen"] is False
    assert found["warm"]["selection_frozen"] is True


def test_removing_something_absent_is_refused(tmp_path: Path):
    mission.create(tmp_path, mission_id="absent", name="x", scrolls=["PHerc0001"])
    with pytest.raises(mission.MissionError, match="none of those"):
        mission.remove_scrolls(tmp_path / "absent", remove=["PHerc9999"], reason="typo")


def test_a_queued_job_freezes_the_selection_before_its_receipt_exists(tmp_path: Path):
    """The panel's mission listing treats a queued job as work; `has_work`
    alone reads receipts on disk and cannot see a job that has not finished
    yet. `frozen_by_queue` is the caller's way of saying what the queue
    already knows, so the two do not disagree about whether an edit needs a
    reason.
    """
    mission.create(tmp_path, mission_id="racing", name="x", scrolls=["PHerc0001"])
    directory = tmp_path / "racing"
    assert mission.has_work(directory) is False

    with pytest.raises(mission.MissionError, match="frozen"):
        mission.amend_scrolls(directory, add=["PHerc0002"], frozen_by_queue=True)
    amended = mission.amend_scrolls(directory, add=["PHerc0002"], reason="a job is running",
                                    frozen_by_queue=True)
    assert amended["amendments"][-1]["reason"] == "a job is running"

    with pytest.raises(mission.MissionError, match="frozen"):
        mission.remove_scrolls(directory, remove=["PHerc0002"], frozen_by_queue=True)
    narrowed = mission.remove_scrolls(directory, remove=["PHerc0002"], reason="wrong scroll",
                                      frozen_by_queue=True)
    assert narrowed["amendments"][-1]["reason"] == "wrong scroll"
