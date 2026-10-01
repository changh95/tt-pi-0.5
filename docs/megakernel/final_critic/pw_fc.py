import json, time
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(headless=True, args=["--autoplay-policy=no-user-gesture-required"])
    pg = b.new_page(viewport={"width": 1280, "height": 2000})
    pg.goto("https://huggingface.co/changh95/pi05-base-p150", wait_until="networkidle", timeout=90000)
    v = pg.locator("video")
    print("videos", v.count())
    v.first.scroll_into_view_if_needed()
    st = pg.evaluate("""async () => { const v=document.querySelector('video'); v.muted=true; v.currentTime=12; await v.play();
      await new Promise(r=>setTimeout(r,4000)); return {err: v.error && v.error.code, ready: v.readyState, t: v.currentTime,
      dur: v.duration, paused: v.paused, w: v.videoWidth, h: v.videoHeight, src: v.currentSrc}; }""")
    print("STATE", json.dumps(st))
    v.first.screenshot(path="video_frame.png")
    txt = pg.inner_text("body")
    print("page has 55.84", "55.84" in txt, "whole-model", "ONE persistent" in txt or "ONE fused op" in txt)
    b.close()
