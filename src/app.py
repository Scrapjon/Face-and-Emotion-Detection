"""
app.py
------
Tkinter UI for the facial recognition pipeline.

Changes in this version:
  - Side panel showing the closest-match reference image for every detected face.
    Multiple simultaneous detections are each shown as their own card.
  - Register-unknown flow: when an UNKNOWN face is detected, a modal dialog
    asks for a name. The dialog shows a preview of the face crop. Names are
    normalised to lowercase so registration is fully case-insensitive.
    The image is saved with a UUID filename to prevent collisions.
    Multiple unknowns are queued and handled one at a time so dialogs
    never stack.
  - "Add Face" button: manually register any currently detected face,
    regardless of whether it is already known.
"""

import cv2
import tkinter as tk
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from PIL import Image, ImageTk, ImageDraw
from threading import Thread, Lock
import uuid

from face_emotion.face_recognition import FacialRecognitionModel

# Constants

WINDOW_TITLE  = "Face and Emotion Detection"
FEED_SIZE     = (600, 600)
PANEL_WIDTH   = 260
THUMB_SIZE    = (220, 180)
CARD_PAD      = 10
UNKNOWN_NAME  = "UNKOWN"

PANEL_BG  = "#111827"
CARD_BG   = "#1f2937"
HEADER_BG = "#0f3460"
TEXT_FG   = "#e5e7eb"
SUB_FG    = "#9ca3af"

# Monkey-patches applied before App is constructed

def _patch_model():
    original_identify = FacialRecognitionModel._identify

    def _identify_cached(self, embedding):
        name, score = original_identify(self, embedding)
        self._last_score = score
        return name, score

    FacialRecognitionModel._identify = _identify_cached

    original_detect = FacialRecognitionModel.detect

    def _detect_with_extras(self, frame):
        self._last_score = 0.0
        results = original_detect(self, frame)
        if frame is None or frame.size == 0:
            return results
        for r in results:
            x, y, w, h = r["box"]
            r["face_crop"] = frame[max(0, y): y + h, max(0, x): x + w].copy()
            if "best_score" not in r:
                r["best_score"] = getattr(self, "_last_score", 0.0)
        return results

    FacialRecognitionModel.detect = _detect_with_extras


_patch_model()


def _first_image(person_dir: Path) -> Optional[Image.Image]:
    if not person_dir.is_dir():
        return None
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for p in sorted(person_dir.glob(ext)):
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                continue
    return None


def _make_card(name: str, score: float, ref_img: Optional[Image.Image], width: int) -> Image.Image:
    card_h = THUMB_SIZE[1] + 52
    card = Image.new("RGB", (width, card_h), CARD_BG)
    draw = ImageDraw.Draw(card)

    if ref_img is not None:
        thumb = ref_img.copy()
        thumb.thumbnail(THUMB_SIZE, Image.LANCZOS)
        x_off = (width - thumb.width) // 2
        card.paste(thumb, (x_off, 4))
    else:
        draw.rectangle([4, 4, width - 4, THUMB_SIZE[1] + 4], fill="#374151")
        draw.text((width // 2, THUMB_SIZE[1] // 2 + 4), "No image", fill=SUB_FG, anchor="mm")

    y = THUMB_SIZE[1] + 10
    draw.text((8, y),      f"{name}", fill=TEXT_FG)
    draw.text((8, y + 20), f"score {score:.2f}", fill=SUB_FG)
    return card


def _build_panel(recognitions: List[dict], db_path: Path, width: int, height: int) -> Image.Image:
    panel = Image.new("RGB", (width, height), PANEL_BG)
    draw  = ImageDraw.Draw(panel)

    draw.rectangle([0, 0, width, 34], fill=HEADER_BG)
    draw.text((width // 2, 17), "Closest Matches", fill=TEXT_FG, anchor="mm")

    y = 40
    card_w = width - 12

    for i, r in enumerate(recognitions):
        name  = r.get("name", UNKNOWN_NAME)
        score = r.get("best_score", 0.0)

        if name in (UNKNOWN_NAME, "SPOOF / FAKE FACE"):
            ref_img = None
        else:
            ref_img = _first_image(db_path / name.lower())

        card = _make_card(name, score, ref_img, card_w)
        panel.paste(card, (6, y))
        y += card.height + CARD_PAD

        if y + 60 > height:
            remaining = len(recognitions) - (i + 1)
            if remaining > 0:
                draw.text((width // 2, y + 8), f"+ {remaining} more", fill=SUB_FG, anchor="mm")
            break

    if not recognitions:
        draw.text((width // 2, height // 2), "No faces detected", fill=SUB_FG, anchor="mm")

    return panel


class App:

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.model   = FacialRecognitionModel(self.db_path)

        self._lock = Lock()
        self._pending_feed:  Optional[ImageTk.PhotoImage] = None
        self._pending_panel: Optional[ImageTk.PhotoImage] = None
        self._current_feed:  Optional[ImageTk.PhotoImage] = None
        self._current_panel: Optional[ImageTk.PhotoImage] = None

        self._latest_recognitions: List[dict] = []

        self._queue: List[dict] = []
        self._queue_lock  = Lock()
        self._dialog_open = False

    def _build_window(self):
        self.root = tk.Tk()
        self.root.title(WINDOW_TITLE)
        self.root.configure(bg=PANEL_BG)
        self.root.resizable(False, False)

        self.feed_label = tk.Label(self.root, bg="black", bd=0)
        self.feed_label.grid(row=0, column=0, sticky="nsew")

        self.panel_label = tk.Label(self.root, bg=PANEL_BG, bd=0)
        self.panel_label.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        add_btn = tk.Button(
            self.root,
            text="Add Face",
            command=self._on_add_face,
            bg=HEADER_BG,
            fg=TEXT_FG,
            activebackground="#1a4a80",
            activeforeground=TEXT_FG,
            relief="flat",
            padx=10,
            pady=6,
        )
        add_btn.grid(row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=(4, 6))

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, minsize=PANEL_WIDTH)

    def _detection_loop(self):
        for frame in self.model.run_stream():
            if not isinstance(frame, np.ndarray):
                continue

            recognitions = list(self.model.prev_recognitions)

            rgb      = frame[:, :, ::-1]
            feed_img = Image.fromarray(rgb).resize(FEED_SIZE)
            feed_ph  = ImageTk.PhotoImage(feed_img)

            panel_img = _build_panel(recognitions, self.db_path, PANEL_WIDTH, FEED_SIZE[1])
            panel_ph  = ImageTk.PhotoImage(panel_img)

            with self._lock:
                self._pending_feed        = feed_ph
                self._pending_panel       = panel_ph
                self._latest_recognitions = recognitions

    def _poll(self):
        with self._lock:
            if self._pending_feed is not None:
                self._current_feed = self._pending_feed
                self._pending_feed = None
                self.feed_label.configure(image=self._current_feed)

            if self._pending_panel is not None:
                self._current_panel = self._pending_panel
                self._pending_panel = None
                self.panel_label.configure(image=self._current_panel)

        self._check_unknowns()
        self.root.after(16, self._poll)

    def _check_unknowns(self):
        if self._dialog_open:
            return

        with self._lock:
            current = list(self._latest_recognitions)

        with self._queue_lock:
            queued_boxes = {item["box"] for item in self._queue}
            for r in current:
                if r.get("name") != UNKNOWN_NAME:
                    continue
                crop = r.get("face_crop")
                box  = r.get("box")
                if crop is None or box in queued_boxes:
                    continue
                self._queue.append({"box": box, "face_crop": crop.copy()})
                queued_boxes.add(box)

            if not self._queue:
                return
            item = self._queue.pop(0)

        self._dialog_open = True
        self._open_register_dialog(item["face_crop"])

    def _on_add_face(self):
        """
        Called when the user clicks Add Face. If exactly one face is currently
        detected, open the register dialog for it. If multiple faces are
        detected, ask the user to pick one by its current label.
        """
        if self._dialog_open:
            return

        with self._lock:
            current = list(self._latest_recognitions)

        crops = [r for r in current if r.get("face_crop") is not None]

        if not crops:
            self._show_info("No face detected in the current frame.")
            return

        if len(crops) == 1:
            self._dialog_open = True
            self._open_register_dialog(crops[0]["face_crop"])
            return

        # Multiple faces: show a picker so the user can choose which to register.
        self._open_face_picker(crops)

    def _open_face_picker(self, recognitions: List[dict]):
        """
        Show a dialog with a thumbnail button for each detected face so the
        user can choose which one to register.
        """
        picker = tk.Toplevel(self.root)
        picker.title("Select Face to Add")
        picker.resizable(False, False)
        picker.grab_set()
        picker.configure(bg=PANEL_BG)

        tk.Label(
            picker,
            text="Multiple faces detected. Choose one to register:",
            bg=PANEL_BG, fg=TEXT_FG,
        ).pack(pady=(12, 8), padx=12)

        btn_frame = tk.Frame(picker, bg=PANEL_BG)
        btn_frame.pack(padx=12, pady=(0, 12))

        # Keep PhotoImage refs alive for the lifetime of the picker.
        picker._photos = []

        def _pick(crop):
            picker.destroy()
            self._dialog_open = True
            self._open_register_dialog(crop)

        for r in recognitions:
            crop  = r["face_crop"]
            label = r.get("name", UNKNOWN_NAME)

            face_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            thumb    = Image.fromarray(face_rgb).resize((100, 100), Image.LANCZOS)
            photo    = ImageTk.PhotoImage(thumb)
            picker._photos.append(photo)

            col = tk.Frame(btn_frame, bg=PANEL_BG)
            col.pack(side="left", padx=6)
            tk.Button(col, image=photo, command=lambda c=crop: _pick(c), bd=0).pack()
            tk.Label(col, text=label, bg=PANEL_BG, fg=SUB_FG, font=("TkDefaultFont", 8)).pack()

        tk.Button(
            picker, text="Cancel",
            command=picker.destroy,
            bg=CARD_BG, fg=TEXT_FG, relief="flat", padx=8, pady=4,
        ).pack(pady=(0, 12))

    def _open_register_dialog(self, face_crop: np.ndarray):
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        preview  = Image.fromarray(face_rgb).resize((160, 160), Image.LANCZOS)
        photo    = ImageTk.PhotoImage(preview)

        dlg = tk.Toplevel(self.root)
        dlg.title("Unknown Person")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.configure(bg=PANEL_BG)
        dlg._photo = photo

        tk.Label(dlg, image=photo, bg=PANEL_BG).pack(pady=(14, 4))
        tk.Label(
            dlg,
            text="Unknown person detected.\nEnter a name to register, or skip.",
            bg=PANEL_BG, fg=TEXT_FG, justify="center",
        ).pack(pady=(0, 8))

        name_var = tk.StringVar()
        entry = tk.Entry(dlg, textvariable=name_var, width=26)
        entry.pack(pady=(0, 10))
        entry.focus_set()

        def _finish(register: bool):
            raw = name_var.get().strip()
            dlg.destroy()
            self._dialog_open = False
            if register and raw:
                self._register(raw.lower(), face_crop)

        btn_row = tk.Frame(dlg, bg=PANEL_BG)
        btn_row.pack(pady=(0, 14))
        tk.Button(btn_row, text="Register", width=12, command=lambda: _finish(True)).pack(side="left", padx=6)
        tk.Button(btn_row, text="Skip",     width=12, command=lambda: _finish(False)).pack(side="right", padx=6)

        dlg.bind("<Return>", lambda _: _finish(True))
        dlg.bind("<Escape>", lambda _: _finish(False))

    def _register(self, name: str, face_crop: np.ndarray):
        person_dir = self.db_path / name
        person_dir.mkdir(parents=True, exist_ok=True)
        img_path = person_dir / f"{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(img_path), face_crop)

        try:
            tensor = self.model._preprocess_face(face_crop)
            emb    = self.model.face_client.forward(tensor)

            if name not in self.model.embedding_db:
                self.model.embedding_db[name] = []
            self.model.embedding_db[name].append(emb)
            self.model._save_db()
            print(f"[register] '{name}' now has {len(self.model.embedding_db[name])} embeddings")
        except Exception as e:
            print(f"[register] embedding failed for '{name}': {e}")

    def _show_info(self, message: str):
        dlg = tk.Toplevel(self.root)
        dlg.title("Info")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.configure(bg=PANEL_BG)
        tk.Label(dlg, text=message, bg=PANEL_BG, fg=TEXT_FG, padx=20, pady=16).pack()
        tk.Button(dlg, text="OK", command=dlg.destroy, width=10).pack(pady=(0, 12))

    def run(self):
        self._build_window()
        Thread(target=self._detection_loop, daemon=True).start()
        self.root.after(16, self._poll)
        self.root.mainloop()
        self.model.stop_stream()
        print("Exiting…")


if __name__ == "__main__":
    app = App("debug_data")
    app.run()