"""REST: job CRUD + SSE."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from deps import manager
from jobs.models import JobConfig, JobSnapshot

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.get("")
async def list_jobs() -> dict:
    return {"jobs": [j.model_dump() for j in manager.list_jobs()], "queue": manager.queue_status()}


@router.post("")
async def create_job(body: JobConfig) -> dict:
    try:
        manager.dataset_path(body.dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    job = manager.create_job(body)
    return job.snapshot().model_dump()


@router.get("/{job_id}")
async def get_job(job_id: str) -> dict:
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job.snapshot().model_dump()


@router.get("/{job_id}/logs")
async def get_job_logs(job_id: str, offset: int = 0, limit: int = 200) -> dict:
    try:
        return manager.read_logs(job_id, offset=offset, limit=limit)
    except KeyError:
        raise HTTPException(404, "Job not found") from None


@router.get("/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    if not manager.get_job(job_id):
        raise HTTPException(404, "Job not found")

    async def gen():
        async for chunk in manager.stream_events(job_id):
            yield chunk

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    try:
        job = manager.cancel_job(job_id)
    except KeyError:
        raise HTTPException(404, "Job not found") from None
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return job.snapshot().model_dump()
