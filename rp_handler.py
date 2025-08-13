import asyncio
import os
import runpod
import json

from app.livekit_worker import Session

# Payload schema (example):
# {
#   "input": {
#     "livekit_url": "wss://your.livekit.server",
#     "token": "<join token>",
#     "driver_identity": "caller-123",        # or null
#     "track_sid": null,                      # optional precise track
#     "source_image_b64": "<base64>",
#     "width": 512, "height": 512, "fps": 24,
#     "backend": "torch"                      # "torch" or "trt"
#   }
# }

async def run_session(job):
    p = job["input"]
    sess = Session(
        url=p["livekit_url"],
        token=p["token"],
        driver_identity=p.get("driver_identity"),
        track_sid=p.get("track_sid"),
        out_width=int(p.get("width", 512)),
        out_height=int(p.get("height", 512)),
        fps=int(p.get("fps", 24)),
        backend=p.get("backend", "torch"),
    )
    yield {"status": "connecting"}
    try:
        await sess.start(p["source_image_b64"])
    except Exception as e:
        yield {"status": "error", "message": str(e)}
        raise
    finally:
        yield {"status": "ended"}

def generator_handler(job):
    # Wrap the asyncio task so RunPod can stream logs/chunks
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async def _run():
        async for chunk in run_session(job):
            yield chunk
    gen = _run()

    # Drive the async generator synchronously for RunPod
    while True:
        try:
            nxt = loop.run_until_complete(gen.__anext__())
            yield nxt
        except StopAsyncIteration:
            break

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": generator_handler,
        "return_aggregate_stream": True  # stream results also available at /run
    })
