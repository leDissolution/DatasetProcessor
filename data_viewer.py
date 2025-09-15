#!/usr/bin/env python3
"""
JSONL Data Viewer - A simple GUI to browse through JSONL entries with critical token highlighting
"""
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os

# Hide console window on Windows
if os.name == 'nt':
    import ctypes
    ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

# --- Tokenizer setup ---
try:
    from transformers import AutoTokenizer
    TOKENIZER_PATH = r"I:\StatSuite\Artifacts\g2b-stage-0"
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
except Exception as e:
    tokenizer = None
    print(f"[WARNING] Could not load tokenizer: {e}")

class DataViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("JSONL Data Viewer")
        self.root.geometry("1200x800")
        
        self.data = []  # currently displayed data (filtered or not)
        self.all_data = []  # all loaded entries (unfiltered)
        self.flips_only_data = []  # only flips
        self.current_index = 0
        self.show_flips_only = tk.BooleanVar(value=True)
        self.search_var = tk.StringVar()
        self.filtered_data = []  # filtered by search
        self.current_search = ""
        
        self.setup_ui()
        self.bind_keys()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # File selection
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        ttk.Label(file_frame, text="File:").grid(row=0, column=0, padx=(0, 5))
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, state="readonly", width=60).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2)
        
        # Navigation frame
        nav_frame = ttk.Frame(main_frame)
        nav_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        self.prev_btn = ttk.Button(nav_frame, text="Previous", command=self.prev_entry, state="disabled")
        self.prev_btn.grid(row=0, column=0, padx=(0, 5))
        
        self.next_btn = ttk.Button(nav_frame, text="Next", command=self.next_entry, state="disabled")
        self.next_btn.grid(row=0, column=1, padx=(0, 10))
        
        self.counter_var = tk.StringVar(value="0 / 0")
        ttk.Label(nav_frame, textvariable=self.counter_var).grid(row=0, column=2, padx=(0, 10))
        
        ttk.Label(nav_frame, text="Go to:").grid(row=0, column=3, padx=(0, 5))
        self.jump_var = tk.StringVar()
        jump_entry = ttk.Entry(nav_frame, textvariable=self.jump_var, width=10)
        jump_entry.grid(row=0, column=4, padx=(0, 5))
        jump_entry.bind('<Return>', lambda e: self.jump_to_entry())
        
        ttk.Button(nav_frame, text="Jump", command=self.jump_to_entry).grid(row=0, column=5)
        
        # Flips-only checkbox (after nav frame)
        self.flips_checkbox = ttk.Checkbutton(nav_frame, text="Show flips only", variable=self.show_flips_only, command=self.toggle_flips_mode)
        self.flips_checkbox.grid(row=0, column=6, padx=(10, 0))

        # Search field and buttons
        ttk.Label(nav_frame, text="Search:").grid(row=0, column=7, padx=(10, 0))
        search_entry = ttk.Entry(nav_frame, textvariable=self.search_var, width=20)
        search_entry.grid(row=0, column=8, padx=(0, 5))
        search_entry.bind('<Return>', lambda e: self.apply_search())
        ttk.Button(nav_frame, text="Go", command=self.apply_search).grid(row=0, column=9, padx=(0, 2))
        ttk.Button(nav_frame, text="Clear", command=self.clear_search).grid(row=0, column=10)
        
        # ID-only search checkbox
        self.id_only_var = tk.BooleanVar(value=False)
        self.id_only_checkbox = ttk.Checkbutton(nav_frame, text="ID only", variable=self.id_only_var)
        self.id_only_checkbox.grid(row=0, column=11, padx=(10, 0))
        
        # Button to print flips per source file
        ttk.Button(nav_frame, text="Flips per Source File", command=self.print_flips_per_source_file).grid(row=0, column=12, padx=(10, 0))
        
        # Content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(1, weight=1)
        
        # Entry info
        self.info_var = tk.StringVar()
        info_label = ttk.Label(content_frame, textvariable=self.info_var, background="lightblue", padding="5")
        info_label.grid(row=0, column=0, sticky="we", pady=(0, 5))
        
        # Notebook for tabs
        notebook = ttk.Notebook(content_frame)
        notebook.grid(row=1, column=0, sticky="nsew")
        
        # Prompt tab
        prompt_frame = ttk.Frame(notebook)
        notebook.add(prompt_frame, text="Prompt")
        
        prompt_frame.columnconfigure(0, weight=1)
        prompt_frame.rowconfigure(0, weight=1)
        
        self.prompt_text = scrolledtext.ScrolledText(prompt_frame, wrap=tk.WORD, height=15)
        self.prompt_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Critical token info frame
        token_info_frame = ttk.LabelFrame(prompt_frame, text="Critical Token Information", padding="5")
        token_info_frame.grid(row=1, column=0, sticky="we", padx=5, pady=5)
        token_info_frame.columnconfigure(1, weight=1)
        
        ttk.Label(token_info_frame, text="Position:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.token_pos_var = tk.StringVar()
        ttk.Label(token_info_frame, textvariable=self.token_pos_var).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(token_info_frame, text="Expected:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self.token_expected_var = tk.StringVar()
        ttk.Label(token_info_frame, textvariable=self.token_expected_var).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(token_info_frame, text="Actual:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5))
        self.token_actual_var = tk.StringVar()
        ttk.Label(token_info_frame, textvariable=self.token_actual_var).grid(row=2, column=1, sticky=tk.W)
        
        ttk.Label(token_info_frame, text="Loss:").grid(row=3, column=0, sticky=tk.W, padx=(0, 5))
        self.token_loss_var = tk.StringVar()
        ttk.Label(token_info_frame, textvariable=self.token_loss_var).grid(row=3, column=1, sticky=tk.W)
        
        # Metadata tab
        metadata_frame = ttk.Frame(notebook)
        notebook.add(metadata_frame, text="Metadata")
        
        metadata_frame.columnconfigure(0, weight=1)
        metadata_frame.rowconfigure(0, weight=1)
        
        self.metadata_text = scrolledtext.ScrolledText(metadata_frame, wrap=tk.WORD)
        self.metadata_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
    def bind_keys(self):
        self.root.bind('<Left>', lambda e: self.prev_entry())
        self.root.bind('<Right>', lambda e: self.next_entry())
        self.root.bind('<Control-o>', lambda e: self.browse_file())
        
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select JSONL file",
            filetypes=[("JSONL files", "*.jsonl"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            self.load_file(file_path)
            
    def apply_search(self):
        search_text = self.search_var.get().strip().lower()
        self.current_search = search_text
        base_data = self.flips_only_data if self.show_flips_only.get() else self.all_data
        if search_text:
            if self.id_only_var.get():
                # Search only in id (or id field)
                self.filtered_data = [entry for entry in base_data if search_text in str(entry.get('id', '')).lower()]
            else:
                self.filtered_data = [entry for entry in base_data if search_text in json.dumps(entry, ensure_ascii=False).lower()]
        else:
            self.filtered_data = base_data.copy()
        self.data = self.filtered_data
        self.current_index = 0
        if not self.data:
            self.counter_var.set("0 / 0")
            self.update_display()
            messagebox.showinfo("No Results", "No entries match your search.")
        else:
            self.update_display()

    def clear_search(self):
        self.search_var.set("")
        self.current_search = ""
        base_data = self.flips_only_data if self.show_flips_only.get() else self.all_data
        self.filtered_data = base_data.copy()
        self.data = self.filtered_data
        self.current_index = 0
        self.update_display()

    def toggle_flips_mode(self):
        # Switch between flips-only and all data, and re-apply search if needed
        base_data = self.flips_only_data if self.show_flips_only.get() else self.all_data
        if self.current_search:
            search_text = self.current_search
            if self.id_only_var.get():
                self.filtered_data = [entry for entry in base_data if search_text in str(entry.get('id', '')).lower()]
            else:
                self.filtered_data = [entry for entry in base_data if search_text in json.dumps(entry, ensure_ascii=False).lower()]
        else:
            self.filtered_data = base_data.copy()
        self.data = self.filtered_data
        self.current_index = 0
        if not self.data:
            self.counter_var.set("0 / 0")
            self.update_display()
            messagebox.showerror("Error", "No entries to display in this mode.")
        else:
            self.update_display()
            
    def load_file(self, file_path):
        try:
            self.all_data = []
            unique = {}
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            # Use id as unique key
                            key = entry.get('id', None)
                            if key is not None:
                                # Keep only the entry with the highest worst_loss for each id (now under loss_metrics)
                                def _worst(e):
                                    return (e.get('loss_metrics') or {}).get('worst_loss', 0)
                                if key not in unique or (_worst(entry) > _worst(unique[key])):
                                    unique[key] = entry
                            else:
                                # If no id, just add (rare)
                                self.all_data.append(entry)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
            # Only keep deduped entries with id
            self.all_data.extend(unique.values())
            # Prepare flips-only data
            self.flips_only_data = []
            for entry in self.all_data:
                ct = (entry.get('loss_metrics') or {}).get('critical_token') or {}
                pred = ct.get('pred_decoded_value')
                actual = ct.get('decoded_value')
                if pred is not None and actual is not None and str(pred) != str(actual):
                    self.flips_only_data.append(entry)
            # Sort both lists by worst_loss descending (now under loss_metrics)
            def _worst_loss_key(x):
                return (x.get('loss_metrics') or {}).get('worst_loss', 0)
            self.all_data.sort(key=_worst_loss_key, reverse=True)
            self.flips_only_data.sort(key=_worst_loss_key, reverse=True)
            # Set data based on checkbox and search
            base_data = self.flips_only_data if self.show_flips_only.get() else self.all_data
            if self.current_search:
                search_text = self.current_search
                if self.id_only_var.get():
                    self.filtered_data = [entry for entry in base_data if search_text in str(entry.get('id', '')).lower()]
                else:
                    self.filtered_data = [entry for entry in base_data if search_text in json.dumps(entry, ensure_ascii=False).lower()]
            else:
                self.filtered_data = base_data.copy()
            self.data = self.filtered_data
            if not self.data:
                messagebox.showerror("Error", "No valid entries found in file (after filtering)")
                return
            self.file_path_var.set(file_path)
            self.current_index = 0
            self.update_display()
            self.prev_btn.config(state="normal")
            self.next_btn.config(state="normal")
            messagebox.showinfo("Success", f"Loaded {len(self.all_data)} unique id entries ({len(self.flips_only_data)} flips, sorted by worst_loss)")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
            
    def prev_entry(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
            
    def next_entry(self):
        if self.current_index < len(self.data) - 1:
            self.current_index += 1
            self.update_display()
            
    def jump_to_entry(self):
        try:
            target = int(self.jump_var.get()) - 1  # Convert to 0-based index
            if 0 <= target < len(self.data):
                self.current_index = target
                self.update_display()
            else:
                messagebox.showerror("Error", f"Entry number must be between 1 and {len(self.data)}")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
            
    def update_display(self):
        if not self.data:
            return
            
        entry = self.data[self.current_index]
        
        # Update counter
        self.counter_var.set(f"{self.current_index + 1} / {len(self.data)}")
        
        # Update navigation buttons
        self.prev_btn.config(state="normal" if self.current_index > 0 else "disabled")
        self.next_btn.config(state="normal" if self.current_index < len(self.data) - 1 else "disabled")
        
        # Update entry info
        info_parts = [f"Entry {self.current_index + 1}"]
        if 'id' in entry:
            info_parts.append(f"Source: {entry['id']}")
        # Show target summary if available
        tgt = entry.get('target') or {}
        if tgt:
            ent = tgt.get('entity')
            attr = tgt.get('attr')
            subj = tgt.get('subject_id')
            parts = [p for p in [ent, attr] if p]
            target_str = ".".join(parts)
            if subj:
                target_str += f" ({subj})"
            if target_str:
                info_parts.append(f"Target: {target_str}")
        if 'stat' in entry:
            info_parts.append(f"Stat: {entry['stat']}")
        self.info_var.set(" | ".join(info_parts))
        
        # Update prompt with highlighting
        self.update_prompt(entry)
        
        # Update critical token info
        self.update_critical_token_info(entry)
        
        # Update metadata
        self.update_metadata(entry)
        
    def update_prompt(self, entry):
        self.prompt_text.delete(1.0, tk.END)
        prompt = entry.get('input', entry.get('prompt', 'No prompt available'))
        self.prompt_text.insert(tk.END, prompt)
        # Highlight critical token if available and tokenizer is loaded (now under loss_metrics)
        if tokenizer is not None:
            ct = (entry.get('loss_metrics') or {}).get('critical_token') or {}
            pos = ct.get('position')
            if isinstance(pos, int):
                token_pos = pos + 1
                try:
                    encoding = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
                    offsets = encoding['offset_mapping']
                    # Only proceed if offsets is a list
                    if not isinstance(offsets, list):
                        print(f"[WARNING] offset_mapping is not a list: {type(offsets)}")
                        return
                    # Defensive: check if offsets[token_pos] is a tuple/list of two ints
                    if (
                        0 <= token_pos < len(offsets) and
                        isinstance(offsets[token_pos], (list, tuple)) and
                        len(offsets[token_pos]) == 2
                    ):
                        start_char, end_char = offsets[token_pos]
                        start_line = prompt[:start_char].count('\n') + 1
                        start_col = start_char - (prompt.rfind('\n', 0, start_char) + 1 if '\n' in prompt[:start_char] else 0)
                        end_line = prompt[:end_char].count('\n') + 1
                        end_col = end_char - (prompt.rfind('\n', 0, end_char) + 1 if '\n' in prompt[:end_char] else 0)
                        start_idx = f"{start_line}.{start_col}"
                        end_idx = f"{end_line}.{end_col}"
                        self.prompt_text.tag_configure("critical", background="yellow", foreground="red", font=("TkDefaultFont", 10, "bold"))
                        self.prompt_text.tag_add("critical", start_idx, end_idx)
                        self.prompt_text.see(start_idx)
                except Exception as e:
                    print(f"[WARNING] Could not highlight token: {e}")
                
    def update_critical_token_info(self, entry):
        ct = (entry.get('loss_metrics') or {}).get('critical_token') if isinstance(entry, dict) else None
        if ct:
            self.token_pos_var.set(str(ct.get('position', 'N/A')))
            self.token_actual_var.set(f'"{ct.get("pred_decoded_value", "N/A")}" (ID: {ct.get("pred_token_id", "N/A")})')
            self.token_expected_var.set(f'"{ct.get("decoded_value", "N/A")}" (ID: {ct.get("token_id", "N/A")})')
            self.token_loss_var.set(f'{ct.get("loss", 0):.6f}')
        else:
            self.token_pos_var.set("N/A")
            self.token_expected_var.set("N/A")
            self.token_actual_var.set("N/A")
            self.token_loss_var.set("N/A")
            
    def update_metadata(self, entry):
        self.metadata_text.delete(1.0, tk.END)
        
        # Create a formatted display of all metadata
        metadata_lines = []
        
        # Basic identifiers
        for k in ['id', 'id']:
            if k in entry:
                metadata_lines.append(f"{k}: {entry[k]}")
        
        # Target info (new structure)
        tgt = entry.get('target') or {}
        if tgt:
            metadata_lines.append("target:")
            try:
                metadata_lines.append(json.dumps(tgt, indent=2))
            except Exception:
                metadata_lines.append(str(tgt))
        
        # Add loss metrics (new structure)
        lm = entry.get('loss_metrics') or {}
        if lm:
            metrics = ['completion_difficulty', 'mean_loss', 'worst_loss']
            for metric in metrics:
                if metric in lm:
                    value = lm[metric]
                    if isinstance(value, float):
                        metadata_lines.append(f"{metric}: {value:.6f}")
                    else:
                        metadata_lines.append(f"{metric}: {value}")
        
        # Add other legacy fields if present
        other_fields = ['from', 'previous_from', 'previous_message']
        for field in other_fields:
            if field in entry:
                value = entry[field]
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                metadata_lines.append(f"{field}: {value}")
                
        # Add state information
        if 'old_state' in entry:
            metadata_lines.append("Old State:")
            metadata_lines.append(json.dumps(entry['old_state'], indent=2))
            
        if 'new_state' in entry:
            metadata_lines.append("\nNew State:")
            metadata_lines.append(json.dumps(entry['new_state'], indent=2))
            
        self.metadata_text.insert(tk.END, "\n".join(metadata_lines))

    def print_flips_per_source_file(self):
        # Count flips per source file (first part of id before ':')
        from collections import Counter
        counter = Counter()
        for entry in self.flips_only_data:
            id = entry.get('id', '')
            if ':' in id:
                source_file = id.split(':', 1)[0]
            else:
                source_file = id
            counter[source_file] += 1
        print("Flips per source file:")
        for source_file, count in counter.most_common():
            print(f"{source_file}: {count}")
        messagebox.showinfo("Flips per Source File", "Printed to console.")


def main():
    root = tk.Tk()
    app = DataViewer(root)
    
    # If a file is provided as argument, load it
    import sys
    if len(sys.argv) > 1:
        app.load_file(sys.argv[1])
    
    root.mainloop()


if __name__ == "__main__":
    main()
