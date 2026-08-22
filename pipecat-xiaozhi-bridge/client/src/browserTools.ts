/** Built-in browser tools executed when the agent calls browser__* MCP tools. */

type Facing = "environment" | "user";

function resizeJpeg(canvas: HTMLCanvasElement, maxSide = 1280, quality = 0.82): string {
  const w = canvas.width;
  const h = canvas.height;
  const scale = Math.min(1, maxSide / Math.max(w, h));
  const out = document.createElement("canvas");
  out.width = Math.round(w * scale);
  out.height = Math.round(h * scale);
  const ctx = out.getContext("2d");
  if (!ctx) throw new Error("canvas unavailable");
  ctx.drawImage(canvas, 0, 0, out.width, out.height);
  const dataUrl = out.toDataURL("image/jpeg", quality);
  const b64 = dataUrl.split(",", 2)[1] ?? "";
  return b64;
}

/** User-initiated capture from composer (returns data URL). */
export async function capturePhotoDataUrl(facing: Facing = "environment"): Promise<string> {
  const b64 = await capturePhotoBase64(facing);
  return `data:image/jpeg;base64,${b64}`;
}

async function capturePhotoBase64(facing: Facing = "environment"): Promise<string> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Camera not supported in this browser");
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: facing, width: { ideal: 1920 }, height: { ideal: 1080 } },
    audio: false,
  });
  try {
    const video = document.createElement("video");
    video.srcObject = stream;
    video.playsInline = true;
    await video.play();
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("canvas unavailable");
    ctx.drawImage(video, 0, 0);
    return resizeJpeg(canvas);
  } finally {
    stream.getTracks().forEach((t) => t.stop());
  }
}

/** Agent-initiated capture — shows brief overlay so user sees shutter. */
async function captureWithOverlay(facing: Facing): Promise<string> {
  const overlay = document.createElement("div");
  overlay.className = "camera-capture-overlay";
  overlay.innerHTML = `
    <div class="camera-capture-card">
      <p class="camera-capture-title">Agent R — taking photo…</p>
      <video class="camera-capture-video" autoplay playsinline muted></video>
      <p class="camera-capture-hint muted">Allow camera if prompted</p>
    </div>`;
  document.body.appendChild(overlay);
  const video = overlay.querySelector("video") as HTMLVideoElement;
  let stream: MediaStream | null = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: facing },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    await new Promise((r) => setTimeout(r, 600));
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("canvas unavailable");
    ctx.drawImage(video, 0, 0);
    return resizeJpeg(canvas);
  } finally {
    stream?.getTracks().forEach((t) => t.stop());
    overlay.remove();
  }
}

async function listBluetooth(): Promise<unknown> {
  const bt = navigator.bluetooth;
  if (!bt) {
    return { error: "Web Bluetooth not supported", devices: [] };
  }
  const devices = await bt.getDevices();
  return {
    devices: devices.map((d) => ({
      id: d.id,
      name: d.name || "(unnamed)",
      gatt_connected: d.gatt?.connected ?? false,
    })),
    note: "Only previously permitted devices are listed. Full scan needs a user tap in browser.",
  };
}

async function readGeolocation(highAccuracy: boolean): Promise<unknown> {
  if (!navigator.geolocation) {
    return { error: "Geolocation not supported" };
  }
  return new Promise((resolve) => {
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        resolve({
          latitude: pos.coords.latitude,
          longitude: pos.coords.longitude,
          accuracy_m: pos.coords.accuracy,
          altitude: pos.coords.altitude,
          heading: pos.coords.heading,
          speed: pos.coords.speed,
        });
      },
      (err) => resolve({ error: err.message || String(err.code) }),
      { enableHighAccuracy: highAccuracy, timeout: 15000, maximumAge: 60000 }
    );
  });
}

export async function executeBuiltinTool(
  tool: string,
  args: Record<string, unknown>
): Promise<string> {
  switch (tool) {
    case "take_picture": {
      const facing = args.facing === "user" ? "user" : "environment";
      const b64 = await captureWithOverlay(facing);
      return JSON.stringify({
        ok: true,
        mime: "image/jpeg",
        width_hint: "resized max 1280px",
        image_base64: b64,
      });
    }
    case "list_bluetooth_devices":
      return JSON.stringify(await listBluetooth());
    case "get_geolocation":
      return JSON.stringify(await readGeolocation(Boolean(args.high_accuracy)));
    default:
      return JSON.stringify({ error: `unknown browser tool: ${tool}` });
  }
}
