import os
import re
import torch
import pandas as pd
import tkinter as tk
from tkinter import scrolledtext, Frame, Label, Entry, Button
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    BartTokenizer,
    BartForConditionalGeneration
)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
import random
from rapidfuzz import fuzz, process
from datetime import datetime
# -------------------------------
# Paths to models and CSV
# -------------------------------
INTENT_MODEL_PATH = "./intent_results/model"  # DistilBERT intent model
BART_MODEL_PATH = "./bart_model_fast"         # BART fine-tuned model
CSV_PATH = "CustomerSupportTraining.csv"      # Must contain 'instruction', 'response', 'intent'

# -------------------------------
# Load CSV and build intent->response dict
# -------------------------------
data = pd.read_csv(CSV_PATH)
intent_to_response = {}
for intent, group in data.groupby("intent"):
    responses = group["response"].dropna()
    responses = responses[responses.str.strip() != ""]  # remove empty strings
    if len(responses) == 0:
        continue
    longest_idx = responses.str.len().idxmax()
    intent_to_response[intent] = responses.loc[longest_idx]

# -------------------------------
# Load DistilBERT for intent classification
# -------------------------------
intent_tokenizer = DistilBertTokenizerFast.from_pretrained(INTENT_MODEL_PATH)
intent_model = DistilBertForSequenceClassification.from_pretrained(INTENT_MODEL_PATH)
intent_model.eval()

# -------------------------------
# Load BART for response generation
# -------------------------------
bart_tokenizer = BartTokenizer.from_pretrained(BART_MODEL_PATH)
bart_model = BartForConditionalGeneration.from_pretrained(BART_MODEL_PATH)
bart_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
intent_model.to(device)
bart_model.to(device)

# -------------------------------
# Predict intent
# -------------------------------


def predict_intent(text, threshold=0.7):
    enc = intent_tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = intent_model(**enc)
        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

    idx = probs.argmax()
    confidence = float(probs[idx])
    intent = intent_model.config.id2label[idx]

    # Reject low-confidence predictions
    if confidence < threshold:
        return None, confidence
    return intent, confidence



# -------------------------------
# Evaluate BLEU
# -------------------------------
def evaluate_response(user_input, candidate):
    intent, _ = predict_intent(user_input)
    reference_text = intent_to_response.get(intent, "")
    if not reference_text:
        return 0.0
    smoothie = SmoothingFunction().method4
    reference_tokens = [reference_text.split()]
    candidate_tokens = candidate.split()
    return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)


def get_response_template(intent):
    # Prefer JSON if available
    intent_info = intents_data["intents"].get(intent, {})
    responses = intent_info.get("responses", [])
    if responses:
        return random.choice(responses)

    # Fallback to CSV
    if intent in intent_to_response:
        return intent_to_response[intent]

    return None


import json
# Load intents + placeholders JSON
with open("intents.json", "r", encoding="utf-8") as f:
    intents_data = json.load(f)

import re, random

# --- Normalize odd placeholder casing in CSV responses, e.g. {{oRDER NUMBER}} -> {{Order Number}}
PLACEHOLDER_ALIASES = {
    "order number": "Order Number",
    "tracking number": "Tracking Number",
    "invoice number": "Invoice Number",
    "person name": "Person Name",
    "online company portal info": "Online Company Portal Info",
    "online order interaction": "Online Order Interaction",
    "customer support hours": "Customer Support Hours",
    "customer support phone number": "Customer Support Phone Number",
    "website url": "Website URL",
    "company name": "Company Name",
    "account type": "Account Type",
    "account category": "Account Category",
    "money amount": "Money Amount",
    "refund amount": "Refund Amount",   # ✅ NEW
    "contact method": "Contact Method"
}


def normalize_placeholders(text: str) -> str:
    def repl(m):
        inner = m.group(1).strip()
        canon = PLACEHOLDER_ALIASES.get(inner.casefold(), inner)
        return "{{" + canon + "}}"
    # Replace ANY {{ ... }} case-insensitively to canonical names if known
    return re.sub(r"\{\{\s*([^}]+?)\s*\}\}", repl, text)

# Apply normalization to CSV columns we will use
data["response"] = data["response"].astype(str).apply(normalize_placeholders)
data["instruction"] = data["instruction"].astype(str)

# Keep groups for runtime selection and a fallback "longest" dict
intent_groups = {i: g.reset_index(drop=True) for i, g in data.groupby("intent")}
intent_to_response = {}
for intent, group in intent_groups.items():
    resp_series = group["response"].astype(str)
    longest_idx = resp_series.str.len().idxmax()
    intent_to_response[intent] = resp_series.loc[longest_idx]


def fill_placeholders(response_template: str, user_input: str, intent: str) -> str:
    if not response_template:
        return ""

    response_template = normalize_placeholders(response_template)

    # --- Extract values from intent-specific rules in JSON
    intent_info = intents_data["intents"].get(intent, {})
    intent_placeholder_rules = intent_info.get("placeholders", {})

    for name, pattern in intent_placeholder_rules.items():
        match = re.search(pattern, user_input, flags=re.IGNORECASE)
        if match:
            value = match.group(0)
            response_template = re.sub(
                r"\{\{\s*" + re.escape(name) + r"\s*\}\}",
                value,
                response_template,
                flags=re.IGNORECASE
            )

    # --- Global placeholders (with dynamic detection)
    global_ph = intents_data.get("placeholders", {})
    for placeholder, options in global_ph.items():
        if placeholder not in response_template:
            continue

        replacement = None

        if options and options[0] != "<dynamic>":
            # Predefined replacements
            replacement = random.choice(options)
        else:
            # Handle dynamic placeholders by regex
            if placeholder == "{{Order Number}}":
                match = re.search(r"\b\d{3,}\b", user_input)
            elif placeholder == "{{Tracking Number}}":
                match = re.search(r"\b[A-Z0-9]{8,}\b", user_input)
            elif placeholder == "{{Invoice Number}}":
                match = re.search(r"\b\d{5,}\b", user_input)
            elif placeholder in ["{{Refund Amount}}", "{{Money Amount}}"]:
                match = re.search(r"(RM\s?\d+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?\s?(?:USD|MYR|dollars))", user_input, re.IGNORECASE)
            elif placeholder == "{{Account Type}}":
                match = re.search(r"\b(saving|current|premium)\b", user_input, re.IGNORECASE)
            elif placeholder == "{{Account Category}}":
                match = re.search(r"\b(personal|business|corporate)\b", user_input, re.IGNORECASE)
            elif placeholder == "{{Person Name}}":
                # Match names: one or more capitalized words (e.g., John, John Doe, Mary Ann Lee)
                match = re.search(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b", user_input)
            elif placeholder == "{{Client Last Name}}":
                # Find all capitalized words and take the last one
                matches = re.findall(r"\b[A-Z][a-z]+\b", user_input)
                match = matches[-1] if matches else None
            else:
                match = None

            if isinstance(match, str):
                replacement = match
            elif match:
                replacement = match.group(0)
            else:
                replacement = "<dynamic>"


        response_template = response_template.replace(placeholder, replacement)

    return response_template




from rapidfuzz import fuzz, process
import random
# -------------------------------
# Generate best BART response
# -------------------------------

def normalize_instruction(text: str) -> str:
    # Replace any {{...}} placeholder with a neutral token
    return re.sub(r"\{\{[^}]+?\}\}", "__PH__", text)


def choose_best_response_for_intent(intent: str, user_input: str) -> str | None:
    group = intent_groups.get(intent)
    if group is None or group.empty:
        return None

    # Normalize user input (replace numbers with __PH__)
    user_norm = re.sub(r"\b\d{3,}\b", "__PH__", user_input)

    best_score = -1
    best_row_idx = None
    for i, inst in enumerate(group["instruction"].fillna("").astype(str)):
        inst_norm = normalize_instruction(inst)
        score = fuzz.token_set_ratio(user_norm, inst_norm)
        if score > best_score:
            best_score = score
            best_row_idx = i

    if best_row_idx is not None and best_score >= 65:  # lowered threshold
        return group.loc[best_row_idx, "response"]

    return intent_to_response.get(intent)



greeting_responses = [
    "Hello! How can I help you today?",
    "Hi there 👋, what can I do for you?",
    "Hey! Need any assistance?",
    "Welcome back! How may I assist you?",
    "Good to see you! How can I support you today?",
    "Hi 👋, what brings you here today?",
    "Hello! Need help with something?",
    "Hey there! How can I make your day easier?",
    "Hi, I’m here to help! What do you need?",
    "Welcome! How can I assist you right now?",
    "Hi there, looking for some support?",
    "Hello! What can I do for you today?"
]

exit_responses = [
        "👋 Thanks for chatting with me! Have a great day.",
        "See you soon! Take care 🙌",
        "Goodbye! Wishing you a wonderful day ahead 🌟",
        "Bye for now! Feel free to reach out anytime 💬",
        "Take care! I’ll be here if you need more help 😊",
        "👋 Have a great one! Looking forward to chatting again.",
        "Goodbye! Stay safe and well 🌿",
        "Catch you later! 👋",
        "It was nice talking with you. Have a great day 🌞",
        "Thanks for your time! Goodbye 🙏"
]

def get_bot_response(user_input: str) -> str:
    user_input_lower = user_input.lower().strip()

    # Greetings fast-path
    if any(g in user_input_lower for g in [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "hiya", "yo", "sup", "what's up", "howdy", "greetings", "morning", "afternoon", "evening"
    ]):
        return random.choice(greeting_responses)

    # Exit fast-path
    if any(g in user_input_lower for g in [
        "quit", "q", "bye", "exit", "goodbye", "see you", "see ya", "later",
        "talk to you later", "take care", "farewell", "catch you later", "close"
    ]):
        return random.choice(exit_responses)

    # Ignore very short inputs
    if len(user_input_lower) < 10:
        return "Could you provide a bit more detail?"

    # 1) Exact pattern match from JSON (rare, but cheap)
    for intent_name, intent_info in intents_data["intents"].items():
        for p in intent_info.get("patterns", []):
            if p.lower() == user_input_lower:
                tmpl = choose_best_response_for_intent(intent_name, user_input) or intent_to_response.get(intent_name)
                return fill_placeholders(tmpl, user_input, intent_name) if tmpl else "Sorry, I don’t understand your request."


    # 2) ML-based intent prediction
    intent, confidence = predict_intent(user_input)
    if intent is None:
        return "Sorry, I didn’t quite understand that. Could you rephrase?"

    # 3) Choose best CSV response for that intent
    response_template = choose_best_response_for_intent(intent, user_input)
    if not response_template:
        return "Sorry, I don’t understand your request."

    # 4) Fill placeholders
    final_response = fill_placeholders(response_template, user_input, intent)
    return final_response


class AppChatbot:
    def __init__(self, root):
        self.root = root
        self.root.title("💬 Customer Support Chat")
        self.root.geometry("400x700")  # Mobile-like dimensions
        self.root.configure(bg="#007BFF")  # App blue
        self.root.resizable(True, True)

        # Create header
        self.create_header()

        # Create chat area
        self.create_chat_area()

        # Create input area
        self.create_input_area()

        # Welcome message
        self.add_message("👋 Hello! I'm your customer support assistant. How can I help you today?", is_user=False)

    def create_header(self):
        """Create App-like header"""
        header_frame = Frame(self.root, bg="#023e8a", height=60)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)

        # Profile picture placeholder
        profile_frame = Frame(header_frame, bg="#023e8a", width=40, height=40)
        profile_frame.pack(side=tk.LEFT, padx=15, pady=10)
        profile_frame.pack_propagate(False)

        profile_label = Label(profile_frame, text="🤖", font=("Segoe UI", 16),
                              bg="#023e8a", fg="white")
        profile_label.pack(expand=True)

        # Title and status
        title_frame = Frame(header_frame, bg="#023e8a")
        title_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=10)

        title_label = Label(title_frame, text="Customer Support",
                            font=("Segoe UI", 13, "bold"), bg="#023e8a", fg="white")
        title_label.pack(anchor="w")

        status_label = Label(title_frame, text="Online now",
                             font=("Segoe UI", 10), bg="#023e8a", fg="#B8E6B8")
        status_label.pack(anchor="w")

    def create_chat_area(self):
        """Create scrollable chat area"""
        # Main chat container
        self.chat_container = Frame(self.root, bg="#ECE5DD")
        self.chat_container.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Scrollable canvas
        self.canvas = tk.Canvas(self.chat_container, bg="#ECE5DD", highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.chat_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = Frame(self.canvas, bg="#ECE5DD")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Bind mousewheel to canvas
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.root.bind("<MouseWheel>", self._on_mousewheel)

    def create_input_area(self):
        """Create App-like input area"""
        input_frame = Frame(self.root, bg="#F0F0F0", height=70)
        input_frame.pack(fill=tk.X, side=tk.BOTTOM)
        input_frame.pack_propagate(False)

        # Input container
        input_container = Frame(input_frame, bg="#FFFFFF", relief="solid", bd=1)
        input_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Text entry
        self.message_entry = tk.Text(input_container, height=2, font=("Segoe UI", 11),
                                     bg="#FFFFFF", fg="#333333", relief="flat", bd=0,
                                     wrap=tk.WORD, padx=10, pady=8)
        self.message_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Send button
        self.send_button = Button(input_container, text="➤", font=("Segoe UI", 16, "bold"),
                                  bg="#25D366", fg="white", relief="flat", bd=0,
                                  width=3, command=self.send_message, cursor="hand2")
        self.send_button.pack(side=tk.RIGHT, padx=(5, 10), pady=5)

        # Bind Enter key
        self.message_entry.bind("<Return>", self.on_enter)
        self.message_entry.bind("<Shift-Return>", self.on_shift_enter)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def on_enter(self, event):
        """Handle Enter key press"""
        self.send_message()
        return "break"

    def on_shift_enter(self, event):
        """Handle Shift+Enter for new line"""
        return None

    def add_message(self, message, is_user=False):
        """Add a message to the chat with rounded bubbles"""
        # Message container
        msg_container = Frame(self.scrollable_frame, bg="#ECE5DD")
        msg_container.pack(fill=tk.X, padx=10, pady=5)

        # Create canvas for rounded message bubble
        if is_user:
            bubble_canvas = tk.Canvas(msg_container, bg="#ECE5DD", highlightthickness=0)
            bubble_canvas.pack(side=tk.RIGHT, anchor="e", padx=(50, 0))
            bubble_color = "#ADE8F4"
        else:
            bubble_canvas = tk.Canvas(msg_container, bg="#ECE5DD", highlightthickness=0)
            bubble_canvas.pack(side=tk.LEFT, anchor="w", padx=(0, 50))
            bubble_color = "#F1F1F1"

        # Prepare message with timestamp on same line as last word
        timestamp = datetime.now().strftime("%H:%M")

        # Split message into lines to get the last line
        lines = message.split('\n')
        last_line = lines[-1] if lines else ""

        # Create display message with timestamp on last line
        if len(lines) > 1:
            display_message = '\n'.join(lines[:-1]) + '\n' + last_line + f"  {timestamp}"
        else:
            display_message = message + f"  {timestamp}"

        # Create temporary label to measure text size
        temp_label = Label(self.root, text=display_message, font=("Segoe UI", 11),
                           wraplength=250, justify="left")
        temp_label.update_idletasks()

        text_width = min(temp_label.winfo_reqwidth() + 24, 270)  # Max width with padding
        text_height = temp_label.winfo_reqheight() + 16  # Height with padding
        temp_label.destroy()

        # Configure canvas size
        bubble_canvas.configure(width=text_width, height=text_height)

        def draw_rounded_bubble():
            bubble_canvas.delete("all")
            width = text_width
            height = text_height
            radius = 12

            # Draw rounded rectangle with shadow effect
            # Shadow
            shadow_offset = 2
            bubble_canvas.create_oval(shadow_offset, shadow_offset, radius * 2 + shadow_offset,
                                      height + shadow_offset, fill="#D0D0D0", outline="")
            bubble_canvas.create_oval(width - radius * 2 + shadow_offset, shadow_offset,
                                      width + shadow_offset, height + shadow_offset, fill="#D0D0D0", outline="")
            bubble_canvas.create_rectangle(radius + shadow_offset, shadow_offset,
                                           width - radius + shadow_offset, height + shadow_offset, fill="#D0D0D0",
                                           outline="")

            # Main bubble
            bubble_canvas.create_oval(0, 0, radius * 2, height, fill=bubble_color, outline="")
            bubble_canvas.create_oval(width - radius * 2, 0, width, height, fill=bubble_color, outline="")
            bubble_canvas.create_rectangle(radius, 0, width - radius, height, fill=bubble_color, outline="")

            # Add subtle border
            bubble_canvas.create_oval(0, 0, radius * 2, height, fill="", outline="#E0E0E0", width=1)
            bubble_canvas.create_oval(width - radius * 2, 0, width, height, fill="", outline="#E0E0E0", width=1)
            bubble_canvas.create_line(radius, 0, width - radius, 0, fill="#E0E0E0", width=1)
            bubble_canvas.create_line(radius, height - 1, width - radius, height - 1, fill="#E0E0E0", width=1)

        draw_rounded_bubble()

        # Add message text with timestamp
        msg_text = tk.Text(bubble_canvas, font=("Segoe UI", 11), bg=bubble_color,
                           fg="#333333", relief="flat", bd=0, wrap=tk.WORD,
                           cursor="arrow", highlightthickness=0)
        msg_text.place(x=12, y=8, width=text_width - 24, height=text_height - 16)

        # Insert message text
        msg_text.insert("1.0", message)

        # Insert timestamp with different color on same line as last word
        msg_text.insert(tk.END, f"  {timestamp}")

        # Configure timestamp color
        lines = message.splitlines()
        last_line_length = len(lines[-1]) if lines else 0
        line_number = len(lines)
        timestamp_start = f"{line_number}.{last_line_length + 2}"
        timestamp_end = f"{line_number}.end"
        msg_text.tag_add("timestamp", timestamp_start, timestamp_end)
        msg_text.tag_configure("timestamp", foreground="#999999", font=("Segoe UI", 8))

        # Make text widget read-only
        msg_text.configure(state="disabled")

        # Auto-scroll to bottom
        self.root.after(100, self.scroll_to_bottom)

    def scroll_to_bottom(self):
        """Scroll to the bottom of the chat"""
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def send_message(self):
        """Send user message and get bot response"""
        user_input = self.message_entry.get("1.0", tk.END).strip()

        # Don't send if it's just placeholder text or empty
        if not user_input or user_input == "Type a message...":
            return

        # Add user message
        self.add_message(user_input, is_user=True)

        # # Clear input and reset placeholder
        self.message_entry.delete("1.0", tk.END)
        #self.message_entry.insert("1.0", "Type a message...")
        #self.message_entry.configure(fg="#999999")

        # Show typing indicator (optional)
        self.show_typing_indicator()

        # Get bot response (with slight delay for realism)
        self.root.after(1000, lambda: self.process_bot_response(user_input))

    def show_typing_indicator(self):
        """Show typing indicator"""
        typing_msg = "🤖 is typing..."
        typing_container = Frame(self.scrollable_frame, bg="#ECE5DD")
        typing_container.pack(fill=tk.X, padx=10, pady=5)

        typing_frame = Frame(typing_container, bg="#F0F0F0", relief="solid", bd=0)
        typing_frame.pack(side=tk.LEFT, anchor="w", padx=(0, 50))

        typing_label = Label(typing_frame, text=typing_msg, font=("Segoe UI", 11, "italic"),
                             bg="#F0F0F0", fg="#666666", padx=12, pady=8)
        typing_label.pack()

        self.scroll_to_bottom()
        self.typing_frame = typing_container

    def process_bot_response(self, user_input):
        """Process bot response and display"""
        # Remove typing indicator
        if hasattr(self, 'typing_frame'):
            self.typing_frame.destroy()

        # Get bot response
        bot_response = get_bot_response(user_input)

        # Add bot message
        self.add_message(bot_response, is_user=False)


# -------------------------------
# Run the application
# -------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = AppChatbot(root)
    root.mainloop()


