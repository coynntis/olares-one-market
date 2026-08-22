"""WebSocket live sessions for geometry_preview."""

from __future__ import annotations

import base64
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from deps import manager

router = APIRouter(prefix="/api/v1/live", tags=["live"])


@router.post("/sessions")
async def start_session(mode: str = "geometry_preview") -> dict:
    sess = manager.create_live_session(mode)
    return {
        **sess,
        "ws_url": f"/api/v1/live/{sess['id']}",
        "note": "Connect via WebSocket; send JSON frames or binary JPEG chunks.",
    }


@router.websocket("/{session_id}")
async def live_ws(websocket: WebSocket, session_id: str) -> None:
    sess = manager.get_live_session(session_id)
    if not sess:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if "bytes" in msg and msg["bytes"]:
                sess["frames"] = int(sess.get("frames", 0)) + 1
                await websocket.send_json(
                    {
                        "type": "geometry_preview",
                        "frames": sess["frames"],
                        "message": "chunk received (preview pipeline would run here)",
                    }
                )
            elif "text" in msg and msg["text"]:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    data = {"raw": msg["text"]}
                if data.get("type") == "frame_base64":
                    sess["frames"] = int(sess.get("frames", 0)) + 1
                    raw = base64.b64decode(data.get("data", ""), validate=False)
                    await websocket.send_json(
                        {
                            "type": "geometry_preview",
                            "frames": sess["frames"],
                            "bytes": len(raw),
                            "preview_url": f"/api/v1/live/{session_id}/preview",
                        }
                    )
                else:
                    await websocket.send_json({"type": "ack", "received": data})
    except WebSocketDisconnect:
        pass
