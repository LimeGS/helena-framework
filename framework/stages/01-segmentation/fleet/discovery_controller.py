from __future__ import annotations

from typing import Any

from .discovery_worker import FirstLettersDiscoveryWorker


class FirstLettersDiscoveryController:
    """Server-owned Task 6 mode gate for dedicated noncanonical work."""

    def __init__(
        self, *, mode: str, store: Any, provider: Any,
        pre_run_attempts: int = 1,
        heartbeat_interval_seconds: float | None = None,
    ):
        if mode not in {"off", "shadow", "select"}:
            raise ValueError("unsupported First Letters discovery mode")
        self.mode = mode
        self.store = store
        self.provider = provider
        self.pre_run_attempts = pre_run_attempts
        self.heartbeat_interval_seconds = heartbeat_interval_seconds

    @staticmethod
    def _task9_dormant() -> RuntimeError:
        return RuntimeError("TASK9_DISCOVERY_SELECT_AND_PROMOTION_DORMANT")

    def promote_discovery_evidence(self, *, evidence_set_id: str) -> None:
        del evidence_set_id
        raise self._task9_dormant()

    @staticmethod
    def _off_result() -> dict[str, Any]:
        return {
            "schema": "campaignx.first_letters_discovery_controller_result.v1",
            "mode": "off", "state": "OFF_UNCHANGED", "run_id": None,
            "evidence_set_id": None, "canonical_admission": "PROHIBITED",
        }

    def reserve_baseline_shadow(
        self, *, request_id: str, budget_admission_sha256: str,
    ) -> dict[str, Any]:
        if self.mode == "off":
            return self._off_result()
        if self.mode == "select":
            raise self._task9_dormant()
        if self.store is None:
            raise RuntimeError("shadow discovery requires server store")
        return self.store.reserve_first_letters_baseline_shadow(
            request_id=request_id,
            budget_admission_sha256=budget_admission_sha256,
        )

    def reserve_alternative_shadow(
        self, *, request_id: str, budget_admission_sha256: str,
        arm_id: str,
    ) -> dict[str, Any]:
        if self.mode == "off":
            return self._off_result()
        if self.mode == "select":
            raise self._task9_dormant()
        if self.store is None:
            raise RuntimeError("shadow discovery requires server store")
        return self.store.reserve_first_letters_alternative_shadow(
            request_id=request_id,
            budget_admission_sha256=budget_admission_sha256,
            arm_id=arm_id,
        )

    def run_job(
        self, *, job_id: str, lease_seconds: int,
    ) -> dict[str, Any]:
        if self.mode == "off":
            return self._off_result()
        if self.mode == "select":
            raise self._task9_dormant()
        if self.store is None or self.provider is None:
            raise RuntimeError("shadow discovery requires server store and provider")
        worker = FirstLettersDiscoveryWorker(
            store=self.store, provider=self.provider,
            pre_run_attempts=self.pre_run_attempts,
            heartbeat_interval_seconds=self.heartbeat_interval_seconds,
        )
        return worker.run_job(job_id=job_id, lease_seconds=lease_seconds)

    def run_item(
        self, *, lease_seconds: int, reservation_id: str, item_id: str,
        profile_bytes: bytes,
    ) -> dict[str, Any]:
        if self.mode == "off":
            return self._off_result()
        if self.mode == "select":
            raise self._task9_dormant()
        raise ValueError("DISCOVERY_JOB_ID_REQUIRED")
