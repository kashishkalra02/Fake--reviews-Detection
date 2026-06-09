import tkinter as tk
from PIL import Image, ImageTk, ImageSequence
import pyttsx3
import pandas as pd
import speech_recognition as sr
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# ---------------- LOAD DATA ----------------
df = pd.read_csv("fake reviews dataset.csv")

print("Original Labels:\n", df["label"].value_counts())

# ---------------- FIX LABELS ----------------
df = df[df["label"].isin(["CG", "OR"])]

df["label"] = df["label"].map({
    "CG": 0,   # Fake
    "OR": 1    # Real
})

# ---------------- CLEAN TEXT ----------------
df = df.dropna(subset=["text_"])
df["text_"] = df["text_"].astype(str)
df = df[df["text_"].str.strip() != ""]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["text_"] = df["text_"].apply(clean_text)

# Remove weak/short reviews (boost accuracy)
df = df[df["text_"].str.len() > 20]

print("Cleaned rows:", len(df))

# ---------------- FEATURES ----------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 3),
    min_df=2
)

X = df["text_"]
y = df["label"]

X_vec = vectorizer.fit_transform(X)

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- MODEL (BEST) ----------------
model = LinearSVC(class_weight="balanced")
model.fit(X_train, y_train)

# ---------------- ACCURACY ----------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("High Accuracy:", accuracy)

# ---------------- VOICE ----------------
engine = pyttsx3.init()

def speak(text):
    engine.stop()
    engine.say(text)
    engine.runAndWait()

# ---------------- VOICE INPUT ----------------
recognizer = sr.Recognizer()
mic = sr.Microphone()

def listen_voice():
    with mic as source:
        result_label.config(text="Listening...", fg="yellow")
        root.update()
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        entry.delete("1.0", tk.END)
        entry.insert(tk.END, text)
        check_review(text)
    except:
        speak("Sorry, I could not understand")
        result_label.config(text="Could not understand", fg="orange")

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Fake Review Detector AI")

root.attributes("-fullscreen", True)
root.configure(bg="black")
root.bind("<Escape>", lambda e: root.destroy())

# ---------------- GIF ----------------
gif_path = r"C:\Users\Admin\Desktop\robot _app.py\robot.gif\robot_waves.gif.gif"

gif = Image.open(gif_path)
frames = [ImageTk.PhotoImage(frame.resize((400, 400)))
          for frame in ImageSequence.Iterator(gif)]

gif_label = tk.Label(root, bg="black")
gif_label.pack(pady=20)

frame_index = 0
def update_gif():
    global frame_index
    gif_label.config(image=frames[frame_index])
    frame_index = (frame_index + 1) % len(frames)
    root.after(100, update_gif)

update_gif()

# ---------------- INPUT ----------------
entry = tk.Text(root, height=4, width=80, font=("Arial", 16))
entry.pack(pady=20)

# ---------------- RESULT ----------------
result_label = tk.Label(root, text="", font=("Arial", 24, "bold"),
                        fg="white", bg="black")
result_label.pack(pady=10)

# ---------------- ACCURACY DISPLAY ----------------
accuracy_label = tk.Label(
    root,
    text=f"Accuracy: {accuracy*100:.2f}%",
    font=("Arial", 24, "bold"),
    fg="cyan",
    bg="black"
)
accuracy_label.pack(pady=5)

# ---------------- CHECK REVIEW ----------------
def check_review(text=None):
    if text is None:
        text = entry.get("1.0", tk.END).strip()

    if text == "":
        speak("Please enter or say a review")
        return

    text = clean_text(text)
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]

    if prediction == 1:
        result = "THIS IS A REAL REVIEW"
        result_label.config(text=result, fg="lime")
    else:
        result = "THIS IS A FAKE REVIEW"
        result_label.config(text=result, fg="red")

    speak(result)

# ---------------- BUTTONS ----------------
btn_text = tk.Button(root, text="CHECK TEXT REVIEW", command=check_review,
                     font=("Arial", 20, "bold"),
                     bg="cyan", fg="black")
btn_text.pack(pady=10)

btn_voice = tk.Button(root, text="🎤 SPEAK REVIEW", command=listen_voice,
                      font=("Arial", 20, "bold"),
                      bg="orange", fg="black")
btn_voice.pack(pady=10)

# Speak accuracy once
speak(f"Model accuracy is {accuracy*100:.2f} percent")

root.mainloop()