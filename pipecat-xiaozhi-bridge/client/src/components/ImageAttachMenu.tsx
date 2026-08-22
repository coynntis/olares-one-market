import { useEffect, useRef, useState } from "react";
import { capturePhotoDataUrl } from "../browserTools";
import { IconCamera, IconPaperclip } from "./Icons";

interface Props {
  disabled?: boolean;
  onImage: (dataUrl: string) => void;
  onError: (msg: string) => void;
}

export function ImageAttachMenu({ disabled, onImage, onError }: Props) {
  const [open, setOpen] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const onPickFile = (file: File | null) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      onError("Only image files supported");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      onImage(String(reader.result ?? ""));
      setOpen(false);
    };
    reader.onerror = () => onError("Could not read image file");
    reader.readAsDataURL(file);
  };

  const onTakePhoto = async () => {
    setCapturing(true);
    try {
      const dataUrl = await capturePhotoDataUrl("environment");
      onImage(dataUrl);
      setOpen(false);
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    } finally {
      setCapturing(false);
    }
  };

  return (
    <div className="attach-menu-wrap" ref={rootRef}>
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        className="sr-only"
        onChange={(e) => onPickFile(e.target.files?.[0] ?? null)}
      />
      <button
        type="button"
        className="btn btn-ghost icon-btn"
        title="Image"
        aria-label="Image"
        aria-expanded={open}
        disabled={disabled || capturing}
        onClick={() => setOpen((v) => !v)}
      >
        <IconCamera />
      </button>
      {open && (
        <div className="attach-menu" role="menu">
          <button
            type="button"
            className="attach-menu-item"
            role="menuitem"
            disabled={capturing}
            onClick={() => void onTakePhoto()}
          >
            <IconCamera size={16} />
            <span>{capturing ? "Opening camera…" : "Take photo"}</span>
          </button>
          <button
            type="button"
            className="attach-menu-item"
            role="menuitem"
            onClick={() => {
              fileRef.current?.click();
            }}
          >
            <IconPaperclip size={16} />
            <span>Attach image</span>
          </button>
        </div>
      )}
    </div>
  );
}
