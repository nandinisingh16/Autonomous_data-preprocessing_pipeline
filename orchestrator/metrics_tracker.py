import threading
import json
import os

class PersistentMetricsTracker:
    def __init__(self, storage_file="metrics_store.json"):
        self._lock = threading.RLock()
        self.storage_file = storage_file
        self.load()

    def load(self):
        """Load metrics from file."""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                    self.n_prompts = data.get("prompts", 0)
                    self.n_tasks = data.get("tasks", 0)
                    self.n_corrections = data.get("corrections", 0)
                    self.n_auto_mods = data.get("auto_modifications", 0)
                    self.n_human_mods = data.get("human_modifications", 0)
            else:
                self.reset()
        except Exception as e:
            print(f"[⚠️] Failed to load metrics: {e}")
            self.reset()

    def save(self):
        """Save metrics to file after each increment."""
        with self._lock:
            try:
                data = {
                    "prompts": self.n_prompts,
                    "tasks": self.n_tasks,
                    "corrections": self.n_corrections,
                    "auto_modifications": self.n_auto_mods,
                    "human_modifications": self.n_human_mods,
                }
                with open(self.storage_file, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"[⚠️] Failed to save metrics: {e}")

    def reset(self):
        """Reset all metrics to zero."""
        with self._lock:
            self.n_prompts = 0
            self.n_tasks = 0
            self.n_corrections = 0
            self.n_auto_mods = 0
            self.n_human_mods = 0
            self.save()

    # ---- Counters (now with auto-save) ----
    def task_executed(self):
        with self._lock:
            self.n_tasks += 1
            self.save()

    def prompt_used(self):
        with self._lock:
            self.n_prompts += 1
            self.save()

    def correction_made(self):
        with self._lock:
            self.n_corrections += 1
            self.save()

    def auto_mod(self):
        with self._lock:
            self.n_auto_mods += 1
            self.save()

    def human_mod(self):
        with self._lock:
            self.n_human_mods += 1
            self.save()

    # ---- Computed Metrics ----
    @property
    def PDR(self):
        with self._lock:
            if self.n_tasks == 0:
                return 0.0
            return self.n_prompts / self.n_tasks

    @property
    def SAS(self):
        with self._lock:
            total_mods = self.n_auto_mods + self.n_human_mods
            if total_mods == 0:
                return 0.0
            return self.n_auto_mods / total_mods

    @property
    def COF(self):
        with self._lock:
            if self.n_tasks == 0:
                return 0.0
            return self.n_corrections / self.n_tasks

    @property
    def PTMA(self):
        """PTMA = SAS / (1 + PDR + COF)"""
        pdr = self.PDR
        sas = self.SAS
        cof = self.COF
        denom = 1 + pdr + cof
        return sas / denom if denom != 0 else 0.0

    def to_dict(self):
        """Return all metrics as dict."""
        with self._lock:
            return {
                "prompts": self.n_prompts,
                "tasks": self.n_tasks,
                "corrections": self.n_corrections,
                "auto_modifications": self.n_auto_mods,
                "human_modifications": self.n_human_mods,
                "PDR": round(self.PDR, 4),
                "SAS": round(self.SAS, 4),
                "COF": round(self.COF, 4),
                "PTMA": round(self.PTMA, 4)
            }

# Shared singleton with file persistence
metrics = PersistentMetricsTracker()