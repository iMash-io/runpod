# app/livekit_worker.py
import asyncio
import base64
import os
import time
from typing import Optional

import numpy as np
from livekit import rtc

from .liveportrait_node import LivePortraitNode


def _rgb_to_i420_bytes(rgb: np.ndarray) -> bytes:
    import cv2
    yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)
    return yuv.tobytes()


class Session:
    def __init__(
        self,
        url: str,
        token: str,
        driver_identity: Optional[str],
        track_sid: Optional[str],
        out_width: int,
        out_height: int,
        fps: int,
        backend: str
    ):
        self.url = url
        self.token = token
        self.driver_identity = driver_identity
        self.track_sid = track_sid
        self.out_width = out_width
        self.out_height = out_height
        self.fps = fps
        self.backend = backend
        self.room: Optional[rtc.Room] = None
        self.source = rtc.VideoSource(out_width, out_height)
        self.node = LivePortraitNode(size=min(out_width, out_height), backend=backend)
        self._stop = asyncio.Event()

    async def start(self, source_image_b64: str):
        import cv2
        img_bytes = base64.b64decode(source_image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        src_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Prepare the source portrait once
        self.node.set_source(src_bgr)

        self.room = rtc.Room()
        await self.room.connect(self.url, self.token)

        # publish synthetic video (mark as camera so UIs label it right)
        track = rtc.LocalVideoTrack.create_video_track("liveportrait", self.source)
        pub_opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
        await self.room.local_participant.publish_track(track, pub_opts)

        driver_track = await self._wait_for_driver_video_track()

        video_stream = rtc.VideoStream(driver_track)  # async iterator of frames
        frame_period = 1.0 / float(self.fps)
        next_t = time.perf_counter()

        async for evt in video_stream:
            if self._stop.is_set():
                break

            # Convert driver frame → RGB24 ndarray
            vf = evt.frame  # rtc.VideoFrame
            vf_rgb24 = vf.convert(rtc.VideoBufferType.RGB24)
            rgb = np.frombuffer(vf_rgb24.data, dtype=np.uint8).reshape(
                vf_rgb24.height, vf_rgb24.width, 3
            )

            drv_bgr = rgb[..., ::-1]  # RGB->BGR for OpenCV/LivePortrait

            out_rgb = self.node.animate(drv_bgr)

            if out_rgb.shape[1] != self.out_width or out_rgb.shape[0] != self.out_height:
                out_rgb = cv2.resize(
                    out_rgb,
                    (self.out_width, self.out_height),
                    interpolation=cv2.INTER_AREA
                )

            # Push as I420 to LiveKit
            i420 = _rgb_to_i420_bytes(out_rgb)
            out_vf = rtc.VideoFrame(
                self.out_width, self.out_height, rtc.VideoBufferType.I420, i420
            )
            self.source.capture_frame(out_vf)

            next_t += frame_period
            await asyncio.sleep(max(0, next_t - time.perf_counter()))

        await self._cleanup()

    async def _wait_for_driver_video_track(self) -> rtc.RemoteVideoTrack:
        ready = asyncio.Future()

        def handler_publication(pub: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            if pub.kind == rtc.TrackKind.KIND_VIDEO:
                if self.track_sid and pub.sid != self.track_sid:
                    return
                if self.driver_identity and participant.identity != self.driver_identity:
                    return
                pub.set_subscribed(True)

        def handler_subscribed(track: rtc.Track, pub: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            if track.kind == rtc.TrackKind.KIND_VIDEO and not ready.done():
                ready.set_result(track)

        self.room.on("track_published", handler_publication)
        self.room.on("track_subscribed", handler_subscribed)

        # kick existing pubs
        for p in self.room.remote_participants.values():
            for pub in p.video_tracks.values():
                handler_publication(pub, p)

        remote = await ready
        return remote  # type: ignore

    async def _cleanup(self):
        try:
            if self.room:
                await self.room.disconnect()
        finally:
            self._stop.set()

    def stop(self):
        self._stop.set()
