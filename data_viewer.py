#!/usr/bin/env python3
"""
JSONL Data Viewer - A simple GUI to browse through JSONL entries with critical token highlighting
"""
import json
import math
import statistics
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os

try:
    import numpy as np
except Exception as e:
    np = None
    print(f"[WARNING] NumPy not available: {e}")

# Hide console window on Windows
if os.name == 'nt':
    import ctypes
    ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

# --- Tokenizer setup ---
try:
    from transformers import AutoTokenizer
    TOKENIZER_PATH = r"I:\StatSuite\DockerWorkspace\artifacts\qwen0.6b-stage0"
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
except Exception as e:
    tokenizer = None
    print(f"[WARNING] Could not load tokenizer: {e}")

# --- Plotting setup ---
try:
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None
    print(f"[WARNING] Matplotlib not available: {e}")

class DataViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("JSONL Data Viewer")
        self.root.geometry("1200x800")
        
        self.data = []  # currently displayed data (filtered or not)
        self.all_data = []  # all loaded entries (unfiltered)
        self.raw_data = []  # all entries before id-deduplication
        self.flips_only_data = []  # only flips
        self.entries_by_id = {}  # all versions grouped by id
        self.natural_all_data = []  # original load order (all entries)
        self.natural_flips_only_data = []  # original load order (flips only)
        self.current_index = 0
        self.show_flips_only = tk.BooleanVar(value=True)
        self.search_var = tk.StringVar()
        self.filtered_data = []  # filtered by search
        self.current_search = ""
        self.metric_labels = {
            "worst_loss": "Worst Loss",
            "completion_difficulty": "Completion Difficulty",
            "natural_order": "Natural Order",
        }
        self.metric_keys_by_label = {label: key for key, label in self.metric_labels.items()}
        self.sort_choice = tk.StringVar(value=self.metric_labels["worst_loss"])
        self.mislabel_fix_mode = tk.BooleanVar(value=False)
        self.mislabel_threshold_var = tk.StringVar(value="8")
        self.mislabel_threshold_value = 8.0
        self.difficulty_variance_window = None
        self.difficulty_variance_records = {}
        
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
        
        # Navigation frame with compact rows
        nav_frame = ttk.Frame(main_frame)
        nav_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        nav_frame.columnconfigure(0, weight=1)

        top_row = ttk.Frame(nav_frame)
        top_row.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        top_row.columnconfigure(3, weight=1)

        self.prev_btn = ttk.Button(top_row, text="Previous", command=self.prev_entry, state="disabled")
        self.prev_btn.grid(row=0, column=0, padx=(0, 5))

        self.next_btn = ttk.Button(top_row, text="Next", command=self.next_entry, state="disabled")
        self.next_btn.grid(row=0, column=1, padx=(0, 10))

        self.counter_var = tk.StringVar(value="0 / 0")
        ttk.Label(top_row, textvariable=self.counter_var).grid(row=0, column=2)

        ttk.Frame(top_row).grid(row=0, column=3, sticky="ew")  # spacer

        ttk.Label(top_row, text="Go to:").grid(row=0, column=4, padx=(0, 5))
        self.jump_var = tk.StringVar()
        jump_entry = ttk.Entry(top_row, textvariable=self.jump_var, width=8)
        jump_entry.grid(row=0, column=5, padx=(0, 5))
        jump_entry.bind('<Return>', lambda e: self.jump_to_entry())

        ttk.Button(top_row, text="Jump", command=self.jump_to_entry).grid(row=0, column=6)

        mid_row = ttk.Frame(nav_frame)
        mid_row.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        mid_row.columnconfigure(2, weight=1)

        self.flips_checkbox = ttk.Checkbutton(mid_row, text="Show flips only", variable=self.show_flips_only, command=self.toggle_flips_mode)
        self.flips_checkbox.grid(row=0, column=0, padx=(0, 15))

        ttk.Label(mid_row, text="Search:").grid(row=0, column=1, padx=(0, 5))
        search_entry = ttk.Entry(mid_row, textvariable=self.search_var)
        search_entry.grid(row=0, column=2, padx=(0, 5), sticky="ew")
        search_entry.bind('<Return>', lambda e: self.apply_search())

        ttk.Button(mid_row, text="Go", command=self.apply_search, width=5).grid(row=0, column=3, padx=(0, 2))
        ttk.Button(mid_row, text="Clear", command=self.clear_search, width=6).grid(row=0, column=4, padx=(0, 8))

        self.id_only_var = tk.BooleanVar(value=False)
        self.id_only_checkbox = ttk.Checkbutton(mid_row, text="ID only", variable=self.id_only_var)
        self.id_only_checkbox.grid(row=0, column=5)

        bottom_row = ttk.Frame(nav_frame)
        bottom_row.grid(row=2, column=0, sticky="ew")
        bottom_row.columnconfigure(3, weight=1)

        ttk.Label(bottom_row, text="Sort by:").grid(row=0, column=0, padx=(0, 5))
        self.sort_combobox = ttk.Combobox(
            bottom_row,
            textvariable=self.sort_choice,
            values=list(self.metric_labels.values()),
            state="readonly",
            width=18,
        )
        self.sort_combobox.grid(row=0, column=1, padx=(0, 10))
        self.sort_combobox.bind("<<ComboboxSelected>>", lambda e: self.change_sort_metric())
        self.sort_combobox.set(self.sort_choice.get())

        histogram_menu_btn = ttk.Menubutton(bottom_row, text="Loss Histograms")
        histogram_menu = tk.Menu(histogram_menu_btn, tearoff=0)
        histogram_menu.add_command(label="Deduplicated (by ID)", command=lambda: self.show_loss_histograms(deduplicated=True))
        histogram_menu.add_command(label="All Entries", command=lambda: self.show_loss_histograms(deduplicated=False))
        histogram_menu_btn["menu"] = histogram_menu
        histogram_menu_btn.grid(row=0, column=2, padx=(0, 5))

        ttk.Frame(bottom_row).grid(row=0, column=3, sticky="ew")  # spacer

        ttk.Button(bottom_row, text="Flips per Source File", command=self.print_flips_per_source_file).grid(row=0, column=4, padx=(0, 10))

        ttk.Button(bottom_row, text="Difficulty Variance", command=self.show_difficulty_variance).grid(row=0, column=5, padx=(0, 10))

        self.mislabel_checkbutton = ttk.Checkbutton(
            bottom_row,
            text="Mislabel fix mode",
            variable=self.mislabel_fix_mode,
            command=self.toggle_mislabel_fix_mode,
        )
        self.mislabel_checkbutton.grid(row=0, column=6, padx=(0, 5))

        ttk.Label(bottom_row, text="Threshold:").grid(row=0, column=7, padx=(0, 5))
        self.mislabel_threshold_entry = ttk.Entry(bottom_row, textvariable=self.mislabel_threshold_var, width=6)
        self.mislabel_threshold_entry.grid(row=0, column=8, padx=(0, 5))
        self.mislabel_threshold_entry.bind('<Return>', lambda e: self.apply_mislabel_threshold())
        ttk.Button(bottom_row, text="Apply", command=self.apply_mislabel_threshold).grid(row=0, column=9)
        
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
        if not self.rebuild_active_data():
            messagebox.showinfo("No Results", "No entries match your search.")

    def clear_search(self):
        self.search_var.set("")
        self.current_search = ""
        self.rebuild_active_data()

    def toggle_flips_mode(self):
        if not self.rebuild_active_data():
            messagebox.showerror("Error", "No entries to display in this mode.")

    def toggle_mislabel_fix_mode(self):
        if self.mislabel_fix_mode.get():
            self.flips_checkbox.state(['disabled'])
            self.sort_combobox.config(state="disabled")
            if not self.apply_mislabel_threshold():
                self.mislabel_fix_mode.set(False)
                self.flips_checkbox.state(['!disabled'])
                self.sort_combobox.config(state="readonly")
        else:
            self.flips_checkbox.state(['!disabled'])
            self.sort_combobox.config(state="readonly")
            self.rebuild_active_data()

    def apply_mislabel_threshold(self):
        raw_value = self.mislabel_threshold_var.get().strip()
        try:
            value = float(raw_value)
        except ValueError:
            messagebox.showerror("Invalid Threshold", "Please enter a numeric worst_loss threshold.")
            self.mislabel_threshold_var.set(f"{self.mislabel_threshold_value:g}")
            return False

        self.mislabel_threshold_value = value
        self.mislabel_threshold_var.set(f"{value:g}")
        if self.mislabel_fix_mode.get():
            if not self.rebuild_active_data():
                messagebox.showinfo("No Entries", "No datapoints exceed the current mislabel threshold.")
        return True
            
    def load_file(self, file_path):
        try:
            self.all_data = []
            self.raw_data = []
            self.entries_by_id = {}
            unique = {}
            self.natural_all_data = []
            self.natural_flips_only_data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            self.raw_data.append(entry)
                            # Use id as unique key
                            key = entry.get('id', None)
                            if key is not None:
                                self.entries_by_id.setdefault(key, []).append(entry)
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
            self.natural_all_data = list(self.all_data)
            self.natural_flips_only_data = list(self.flips_only_data)
            self.sort_data()
            self.file_path_var.set(file_path)
            if not self.rebuild_active_data():
                messagebox.showerror("Error", "No valid entries found in file (after filtering)")
                return
            metric_label = self.get_sort_label()
            messagebox.showinfo("Success", f"Loaded {len(self.all_data)} unique id entries ({len(self.flips_only_data)} flips, sorted by {metric_label})")
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
        metric_key = self.get_sort_metric()
        metric_label = self.get_sort_label()
        metric_value = self._get_loss_metric_value(entry, metric_key)
        if metric_value is not None:
            if isinstance(metric_value, float):
                info_parts.append(f"{metric_label}: {metric_value:.6f}")
            else:
                info_parts.append(f"{metric_label}: {metric_value}")
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

    def get_sort_metric(self):
        return self.metric_keys_by_label.get(self.sort_choice.get(), "worst_loss")

    def get_sort_label(self):
        metric = self.get_sort_metric()
        return self.metric_labels.get(metric, metric)

    def _get_loss_metric_value(self, entry, metric):
        if not isinstance(entry, dict):
            return None
        metrics = entry.get('loss_metrics')
        if not isinstance(metrics, dict):
            return None
        return metrics.get(metric)

    def _metric_sort_value(self, entry, metric):
        value = self._get_loss_metric_value(entry, metric)
        if isinstance(value, (int, float)):
            return value
        if value is None:
            return float('-inf')
        try:
            return float(value)
        except (TypeError, ValueError):
            return float('-inf')

    def _get_source_id(self, entry):
        if not isinstance(entry, dict):
            return ""
        raw_id = entry.get('id')
        if raw_id is None:
            return ""
        raw_id = str(raw_id)
        if ':' in raw_id:
            return raw_id.split(':', 1)[0]
        return raw_id

    def _passes_mislabel_threshold(self, entry):
        metric_value = self._get_loss_metric_value(entry, "worst_loss")
        if metric_value is None:
            return False
        try:
            return float(metric_value) > self.mislabel_threshold_value
        except (TypeError, ValueError):
            return False

    def _mislabel_sort_key(self, entry):
        source_id = self._get_source_id(entry).lower()
        worst_loss = self._metric_sort_value(entry, "worst_loss")
        return (source_id, -worst_loss, str(entry.get('id', '')))

    def sort_data(self):
        metric = self.get_sort_metric()
        if metric == "natural_order":
            self.all_data = list(self.natural_all_data)
            self.flips_only_data = list(self.natural_flips_only_data)
            return
        self.all_data.sort(key=lambda x: self._metric_sort_value(x, metric), reverse=True)
        self.flips_only_data.sort(key=lambda x: self._metric_sort_value(x, metric), reverse=True)

    def rebuild_active_data(self, reset_index=True):
        if self.mislabel_fix_mode.get():
            base_data = [entry for entry in self.all_data if self._passes_mislabel_threshold(entry)]
            base_data.sort(key=self._mislabel_sort_key)
        else:
            base_data = self.flips_only_data if self.show_flips_only.get() else self.all_data
        if self.current_search:
            search_text = self.current_search
            if self.id_only_var.get():
                filtered = [entry for entry in base_data if search_text in str(entry.get('id', '')).lower()]
            else:
                filtered = [
                    entry for entry in base_data
                    if search_text in json.dumps(entry, ensure_ascii=False).lower()
                ]
        else:
            filtered = list(base_data)
        self.filtered_data = filtered
        self.data = filtered
        if not self.data:
            self.counter_var.set("0 / 0")
            self.prev_btn.config(state="disabled")
            self.next_btn.config(state="disabled")
            self.update_display()
            return False
        if reset_index:
            self.current_index = 0
        else:
            self.current_index = min(self.current_index, len(self.data) - 1)
        self.update_display()
        return True

    def change_sort_metric(self, *_):
        if not self.all_data and not self.flips_only_data:
            return
        current_id = None
        if self.data and 0 <= self.current_index < len(self.data):
            current_entry = self.data[self.current_index]
            if isinstance(current_entry, dict):
                current_id = current_entry.get('id')
        self.sort_data()
        if not self.rebuild_active_data():
            return
        if current_id is not None:
            for idx, entry in enumerate(self.data):
                if isinstance(entry, dict) and entry.get('id') == current_id:
                    self.current_index = idx
                    self.update_display()
                    break

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

    def show_difficulty_variance(self):
        if not self.entries_by_id:
            messagebox.showinfo("No Data", "Load a dataset before calculating difficulty variance.")
            return

        metric = "completion_difficulty"
        results = {}
        for id_value, variants in self.entries_by_id.items():
            numeric_values = []
            for entry in variants:
                value = self._get_loss_metric_value(entry, metric)
                if value is None:
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(numeric_value):
                    numeric_values.append(numeric_value)
            if len(numeric_values) < 2:
                continue
            mean_value = statistics.mean(numeric_values)
            try:
                variance_value = statistics.variance(numeric_values)
            except statistics.StatisticsError:
                variance_value = 0.0
            std_dev_value = math.sqrt(variance_value) if variance_value >= 0 else float('nan')
            results[id_value] = {
                "count": len(numeric_values),
                "mean": mean_value,
                "variance": variance_value,
                "std_dev": std_dev_value,
                "min": min(numeric_values),
                "max": max(numeric_values),
                "range": max(numeric_values) - min(numeric_values),
                "values": numeric_values,
                "variants": variants,
            }

        if not results:
            messagebox.showinfo("No Variance", "No ids have multiple numeric completion_difficulty values.")
            return

        sorted_items = sorted(results.items(), key=lambda item: item[1]["variance"], reverse=True)
        self.difficulty_variance_records = results

        window_ref = self.difficulty_variance_window
        if window_ref is not None and window_ref.winfo_exists():
            window_ref.destroy()
        self.difficulty_variance_window = None

        window = tk.Toplevel(self.root)
        window.title("Difficulty Variance by ID")
        window.geometry("900x500")
        self.difficulty_variance_window = window
        window.protocol("WM_DELETE_WINDOW", lambda w=window: self._close_difficulty_variance_window(w))

        columns = ("id", "versions", "min", "max", "mean", "variance", "std_dev", "range")
        tree = ttk.Treeview(window, columns=columns, show="headings")
        tree.heading("id", text="ID")
        tree.heading("versions", text="Versions")
        tree.heading("min", text="Min")
        tree.heading("max", text="Max")
        tree.heading("mean", text="Mean")
        tree.heading("variance", text="Variance")
        tree.heading("std_dev", text="Std Dev")
        tree.heading("range", text="Range")

        tree.column("id", width=240, anchor="w")
        tree.column("versions", width=80, anchor="center")
        tree.column("min", width=100, anchor="e")
        tree.column("max", width=100, anchor="e")
        tree.column("mean", width=100, anchor="e")
        tree.column("variance", width=120, anchor="e")
        tree.column("std_dev", width=120, anchor="e")
        tree.column("range", width=100, anchor="e")

        scrollbar = ttk.Scrollbar(window, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)

        for id_value, metrics in sorted_items:
            tree.insert(
                "",
                tk.END,
                values=(
                    id_value,
                    metrics["count"],
                    f"{metrics['min']:.6f}",
                    f"{metrics['max']:.6f}",
                    f"{metrics['mean']:.6f}",
                    f"{metrics['variance']:.6f}",
                    f"{metrics['std_dev']:.6f}",
                    f"{metrics['range']:.6f}",
                ),
            )

        tree.bind("<Double-1>", self.show_difficulty_variance_details)

        ttk.Label(
            window,
            text="Double-click a row to inspect all versions for that id.",
            anchor="w"
        ).grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

    def show_difficulty_variance_details(self, event):
        tree_widget = event.widget
        item_id = tree_widget.identify_row(event.y)
        if not item_id:
            return

        item = tree_widget.item(item_id)
        values = item.get("values", [])
        if not values:
            return
        id_value = values[0]

        # Ensure the row stays selected when the detail window opens
        tree_widget.selection_set(item_id)
        tree_widget.focus(item_id)

        record = self.difficulty_variance_records.get(id_value)
        if not record:
            return

        detail_window = tk.Toplevel(self.root)
        detail_window.title(f"Versions for {id_value}")
        detail_window.geometry("900x600")

        text_widget = scrolledtext.ScrolledText(detail_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True)

        text_widget.insert(tk.END, f"ID: {id_value}\n")
        text_widget.insert(tk.END, f"Versions with numeric completion_difficulty: {record['count']}\n")
        text_widget.insert(tk.END, f"Mean: {record['mean']:.6f}, Variance: {record['variance']:.6f}, Std Dev: {record['std_dev']:.6f}, Range: {record['range']:.6f}\n\n")
        numeric_values_line = ", ".join(f"{value:.6f}" for value in record["values"])
        text_widget.insert(tk.END, f"Values: {numeric_values_line}\n\n")

        for idx, variant in enumerate(record["variants"], 1):
            difficulty = self._get_loss_metric_value(variant, "completion_difficulty")
            worst_loss = self._get_loss_metric_value(variant, "worst_loss")
            text_widget.insert(
                tk.END,
                f"Version {idx}: completion_difficulty={difficulty}, worst_loss={worst_loss}\n"
            )
            try:
                serialized = json.dumps(variant, indent=2, ensure_ascii=False)
            except (TypeError, ValueError):
                serialized = str(variant)
            text_widget.insert(tk.END, serialized + "\n\n")

        text_widget.config(state=tk.DISABLED)

    def _close_difficulty_variance_window(self, window):
        window.destroy()
        if getattr(self, "difficulty_variance_window", None) is window:
            self.difficulty_variance_window = None

    def show_loss_histograms(self, deduplicated=True):
        if plt is None:
            messagebox.showerror("Matplotlib Required", "Matplotlib is not available. Install it to view histograms.")
            return
        if np is None:
            messagebox.showerror("NumPy Required", "NumPy is not available. Install it to view histograms.")
            return
        if not self.all_data:
            messagebox.showerror("No Data", "Load a dataset before viewing histograms.")
            return

        source_data = self.all_data if deduplicated else self.raw_data
        data_label = "Deduplicated" if deduplicated else "All Entries"

        metric_specs = [
            ("worst_loss", self.metric_labels.get("worst_loss", "Worst Loss")),
            ("completion_difficulty", self.metric_labels.get("completion_difficulty", "Completion Difficulty")),
        ]

        plot_data = []
        for key, label in metric_specs:
            values = []
            for entry in source_data:
                value = self._get_loss_metric_value(entry, key)
                if value is None:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(numeric):
                    values.append(numeric)
            if values:
                plot_data.append((label, np.array(values, dtype=float)))

        if not plot_data:
            messagebox.showerror("No Metrics", "No numeric loss metrics available for histogram.")
            return

        fig, axes = plt.subplots(1, len(plot_data), figsize=(6 * len(plot_data), 4))
        if len(plot_data) == 1:
            axes = [axes]

        for ax, (label, values) in zip(axes, plot_data):
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                ax.set_visible(False)
                continue

            data_min = float(finite.min())
            data_max = float(finite.max())
            upper_bound = float(np.percentile(finite, 99))
            if not math.isfinite(upper_bound):
                upper_bound = data_max
            if upper_bound <= data_min:
                upper_bound = data_min + max(abs(data_min) * 0.05, 1e-6)

            # Clip the histogram upper bound to 99th percentile to keep dense shoulders readable.
            clipped = np.clip(finite, data_min, upper_bound)
            bin_count = min(60, max(10, int(math.sqrt(finite.size)) * 2))
            bins = np.linspace(data_min, upper_bound, bin_count + 1)
            counts, _ = np.histogram(clipped, bins=bins)
            positive_counts = counts[counts > 0]
            use_log = bool(positive_counts.size) and counts.max() / positive_counts.min() > 50

            ax.hist(clipped, bins=bins, color="#3b75af", edgecolor="black", alpha=0.7, log=use_log)
            tail_count = int((finite > upper_bound).sum())
            if tail_count:
                ax.text(
                    0.98,
                    0.92,
                    f"{tail_count} values > {upper_bound:.4f}",
                    ha="right",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
                )

            quantile_specs = {
                "Bottom 10%": (np.percentile(finite, 10), "#5a8fc6", (2, 2)),
                "Lower third": (np.percentile(finite, 100 / 3.0), "#1f77b4", (5, 3)),
                "Upper third": (np.percentile(finite, 200 / 3.0), "#1f77b4", (5, 3)),
                "Top 10%": (np.percentile(finite, 90), "#d62728", (2, 1)),
            }

            legend_handles = []
            top_note_y = 0.9
            bottom_note_y = 0.1
            for name, (value, color, dash_pattern) in quantile_specs.items():
                if not math.isfinite(value):
                    continue
                plotted_value = min(max(value, data_min), upper_bound)
                handle = ax.axvline(
                    plotted_value,
                    color=color,
                    linestyle=(0, dash_pattern),
                    linewidth=1.4,
                )
                legend_handles.append((handle, f"{name}: {value:.4f}" + (" (clipped)" if plotted_value != value else "")))
                if plotted_value != value:
                    use_top = "Top" in name or "Upper" in name
                    text_y = top_note_y if use_top else bottom_note_y
                    top_note_y -= 0.08 if use_top else 0.0
                    bottom_note_y += 0.08 if not use_top else 0.0
                    ax.text(
                        0.98,
                        text_y,
                        f"{name} beyond view",
                        ha="right",
                        va="top" if use_top else "bottom",
                        transform=ax.transAxes,
                        fontsize=8,
                        color=color,
                    )

            if legend_handles:
                ax.legend(
                    [h for h, _ in legend_handles],
                    [desc for _, desc in legend_handles],
                    frameon=False,
                    fontsize=8,
                    loc="upper left",
                )

            ax.set_title(label)
            ax.set_xlim(data_min, upper_bound)
            ax.set_xlabel(label)
            ax.set_ylabel("Count (log)" if use_log else "Count")
            ax.grid(axis='y', alpha=0.2)

        fig.suptitle(f"Loss Metric Distributions ({data_label}, n={len(source_data)})")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        plt.show()


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
