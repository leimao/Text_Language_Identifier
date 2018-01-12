import tkinter as tk
from predict import load_model, classify_text

class Todo(tk.Tk):
    def __init__(self, tasks=None):
        super().__init__()


        self.title("Text Language Identifier")
        self.geometry("600x210")

        self.language_note = tk.Label(self, text="Language Identified", bg="lightgrey", fg="black", pady=10)
        self.language_note.pack(side=tk.TOP, fill=tk.X)

        self.language_identified = tk.StringVar()
        self.language_identified.set("")

        self.classification_region = tk.Label(self, textvariable=self.language_identified, bg="grey", fg="white", pady=10)
        self.classification_region.pack(side=tk.TOP, fill=tk.X)

        self.text_region = tk.Text(self, height=10, bg="white", fg="black")

        self.text_region.pack(side=tk.BOTTOM, fill=tk.X)
        self.text_region.focus_set()

        self.bind("<Return>", self.classify_language)

        self.text_region_note = tk.Label(self, text="--- Type or Paste Text Here, and Press Enter ---", bg="lightgrey", fg="black", pady=10)
        self.text_region_note.pack(side=tk.BOTTOM, fill=tk.X)


        self.colour_schemes = [{"bg": "lightgrey", "fg": "black"}, {"bg": "grey", "fg": "white"}]

    def classify_language(self, event=None):
        text_input = self.text_region.get(1.0,tk.END).strip()

        self.language_identified.set(lc[classify_text(text = text_input, model = model, le = le, n_gram_list = n_gram_list)])

        self.text_region.delete(1.0, tk.END)


if __name__ == "__main__":
    model, le, lc, n_gram_list = load_model()
    todo = Todo()
    todo.mainloop()
