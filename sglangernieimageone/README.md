# sglangernieimageone

This app runs SGLang with `baidu/ERNIE-Image` on port `30000`.

The container startup command is:

```bash
git clone https://github.com/sgl-project/sglang.git /workspace/sglang
cd /workspace/sglang
pip install -e "python[diffusion]"
sglang serve --model-path baidu/ERNIE-Image --trust-remote-code --host 0.0.0.0 --port 30000
```

Container image: `lmsysorg/sglang:dev-cu13`

Hugging Face credentials can come from Olares system environment:

```yaml
envs:
  - envName: HF_TOKEN
    valueFrom:
      envName: OLARES_USER_HUGGINGFACE_TOKEN
```

Deployment reads:
- `HF_TOKEN` from `{{ .Values.olaresEnv.HF_TOKEN }}`
- `HF_ENDPOINT` from `{{ .Values.olaresEnv.HF_ENDPOINT }}`

After deployment, send an image generation request:

```bash
curl -X POST http://localhost:30000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "This is a photograph depicting an urban street scene. Shot at eye level, it shows a covered pedestrian or commercial street. Slightly below the center of the frame, a cyclist rides away from the camera toward the background, appearing as a dark silhouette against backlighting with indistinct details. The ground is paved with regular square tiles, bisected by a prominent tactile paving strip running through the scene, whose raised textures are clearly visible under the light. Light streams in diagonally from the right side of the frame, creating a strong backlight effect with a distinct Tyndall effect—visible light beams illuminating dust or vapor in the air and casting long shadows across the street. Several pedestrians appear on the left side and in the distance, some with their backs to the camera and others walking sideways, all rendered as silhouettes or semi-silhouettes. The overall color palette is warm, dominated by golden yellows and dark browns, evoking the atmosphere of dusk or early morning.",
    "height": 1264,
    "width": 848,
    "num_inference_steps": 50,
    "guidance_scale": 4.0,
    "use_pe": true
  }' \
  --output output.png
```
