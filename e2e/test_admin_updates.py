"""Rendered Admin Update view regressions (browser-intercepted API)."""

from collections.abc import Callable

from playwright.sync_api import Page, Route, expect


def _snapshot(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "capable": True,
        "current_version": "5.22.4",
        "latest_version": "5.23.0",
        "update_available": True,
        "repo": "MysticCoss/free-claude-code",
        "branch": "main",
        "auto": False,
        "poll_hours": 6.0,
        "stage": "idle",
        "in_progress": False,
        "message": "",
        "last_check": 1_000_000.0,
        "check_error": None,
        "last_error": None,
        "last_update": None,
        "target_version": None,
    }
    base.update(overrides)
    return base


def _route_json(page: Page, url_glob: str, payload: dict[str, object]) -> None:
    page.route(
        url_glob,
        lambda route: route.fulfill(json=payload),
    )


def _open_update_view(
    page: Page, admin_base_url: str, *, current: str = "5.22.4"
) -> None:
    page.goto(f"{admin_base_url}/admin")
    # Wait for the panel snapshot so the nav button is the only "Update" label.
    expect(page.locator("#updateCurrent")).to_have_text(current)
    page.locator('.nav-link[data-view="updates"]').click()
    expect(page.locator("#view-updates")).to_be_visible()


def test_update_view_shows_versions_fields_and_enabled_update(
    page: Page, admin_base_url: str
) -> None:
    _route_json(page, "**/admin/api/update", _snapshot())
    _open_update_view(page, admin_base_url)

    expect(page.locator("#updateCurrent")).to_have_text("5.22.4")
    expect(page.locator("#updateLatest")).to_have_text("5.23.0")
    expect(page.locator("#updateSource")).to_have_text(
        "MysticCoss/free-claude-code@main"
    )
    expect(page.locator("#updateStage")).to_have_text("idle")
    expect(page.locator("#updateHint")).to_contain_text("FCC 5.23.0 is available")
    expect(page.locator("#updateCheckButton")).to_be_enabled()
    expect(page.locator("#updateApplyButton")).to_be_enabled()
    expect(page.locator("#updateApplyButton")).to_have_text("Update to 5.23.0")

    section = page.locator("#section-updates")
    expect(section).to_be_visible()
    section.get_by_role("button", name="Show advanced", exact=True).click()
    for key in (
        "FCC_UPDATE_REPO",
        "FCC_UPDATE_BRANCH",
        "FCC_UPDATE_AUTO",
        "FCC_UPDATE_POLL_HOURS",
    ):
        expect(section.locator(f"#field-{key}")).to_be_visible()


def test_update_view_disables_actions_for_source_checkout(
    page: Page, admin_base_url: str
) -> None:
    _route_json(
        page,
        "**/admin/api/update",
        _snapshot(
            capable=False,
            current_version="0+unknown",
            latest_version=None,
            update_available=False,
        ),
    )
    _open_update_view(page, admin_base_url, current="0+unknown")

    expect(page.locator("#updateHint")).to_contain_text("source checkout")
    expect(page.locator("#updateCheckButton")).to_be_disabled()
    expect(page.locator("#updateApplyButton")).to_be_disabled()


def test_in_progress_update_blocks_buttons(page: Page, admin_base_url: str) -> None:
    _route_json(
        page,
        "**/admin/api/update",
        _snapshot(
            in_progress=True,
            stage="testing",
            message="Running the pytest gate",
        ),
    )
    _open_update_view(page, admin_base_url)

    expect(page.locator("#updateStage")).to_have_text("Testing")
    expect(page.locator("#updateHint")).to_contain_text(
        "Testing… Running the pytest gate"
    )
    expect(page.locator("#updateApplyButton")).to_be_disabled()
    expect(page.locator("#updateCheckButton")).to_be_disabled()


def test_check_refreshes_the_panel(page: Page, admin_base_url: str) -> None:
    idle = _snapshot(latest_version="5.22.4", update_available=False, check_error=None)
    newer = _snapshot()
    responses: list[Callable[[], dict[str, object]]] = [
        lambda: idle,
        lambda: newer,
    ]

    def handler(route: Route) -> None:
        payload = responses[0]() if route.request.method == "GET" else responses[-1]()
        if route.request.method == "POST":
            responses.reverse()
        route.fulfill(json=payload)

    page.route("**/admin/api/update", handler)
    page.route("**/admin/api/update/check", lambda route: route.fulfill(json=newer))
    _open_update_view(page, admin_base_url)

    expect(page.locator("#updateHint")).to_contain_text("up to date")
    expect(page.locator("#updateApplyButton")).to_be_disabled()

    page.locator("#updateCheckButton").click()

    expect(page.locator("#updateHint")).to_contain_text("5.23.0 is available")
    expect(page.locator("#updateApplyButton")).to_be_enabled()


def test_apply_schedules_update_and_reloads_on_new_instance(
    page: Page, admin_base_url: str
) -> None:
    page.on("dialog", lambda dialog: dialog.accept())
    statuses = iter(
        [
            {"status": "running", "instance_id": "old-instance"},
            {"status": "stopping", "instance_id": "old-instance"},
            {"status": "starting", "instance_id": "old-instance"},
            {"status": "starting", "instance_id": "old-instance"},
            {"status": "starting", "instance_id": "old-instance"},
        ]
    )

    def status_handler(route: Route) -> None:
        try:
            payload = next(statuses)
        except StopIteration:
            payload = {"status": "running", "instance_id": "new-instance"}
        route.fulfill(json=payload)

    _route_json(page, "**/admin/api/update", _snapshot())
    page.route(
        "**/admin/api/update/apply",
        lambda route: route.fulfill(
            json={
                "scheduled": True,
                "from_version": "5.22.4",
                "target_version": "5.23.0",
                "message": "Update started.",
            }
        ),
    )
    page.route("**/admin/api/status", status_handler)
    _open_update_view(page, admin_base_url)

    with page.expect_request("**/admin/api/update/apply") as requested:
        page.locator("#updateApplyButton").click()

    assert requested.value.method == "POST"


def test_apply_error_restores_controls(page: Page, admin_base_url: str) -> None:
    page.on("dialog", lambda dialog: dialog.accept())
    _route_json(page, "**/admin/api/update", _snapshot())
    page.route(
        "**/admin/api/status",
        lambda route: route.fulfill(json={"status": "running", "instance_id": "x"}),
    )
    page.route(
        "**/admin/api/update/apply",
        lambda route: route.fulfill(status=409, json={"detail": "already running"}),
    )
    _open_update_view(page, admin_base_url)

    page.locator("#updateApplyButton").click()

    expect(page.locator("#messageArea")).to_contain_text("already running")
    expect(page.locator("#updateApplyButton")).to_be_enabled()
