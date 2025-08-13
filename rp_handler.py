# rp_handler.py
import asyncio
import base64
import json
import threading
import queue
import runpod

from app.livekit_worker import Session

# ----- async business logic -----
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

# ----- sync handler that streams via a thread-safe queue -----
def handler(job):
    q: "queue.Queue[object | None]" = queue.Queue()

    def runner():
        async def produce():
            try:
                async for chunk in run_session(job):
                    q.put(chunk)
            except Exception as e:
                q.put({"status": "error", "message": str(e)})
            finally:
                q.put(None)

        # run our own loop in this thread (no interference with RunPod's loop)
        asyncio.run(produce())

    threading.Thread(target=runner, daemon=True).start()

    while True:
        item = q.get()
        if item is None:
            break
        yield item

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": True
    })
