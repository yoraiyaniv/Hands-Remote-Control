import tkinter as tk
import subprocess
import threading
import socket
import concurrent.futures

# ── Palette ───────────────────────────────────────────────────────────────────
BG         = "#0a0a0f"
BG2        = "#13131a"
BG3        = "#1c1c26"
BORDER     = "#2a2a3a"
ACCENT     = "#e8c97a"
ACCENT_DIM = "#7a6a3a"
TEXT       = "#e8e8f0"
TEXT_DIM   = "#5a5a70"
GREEN      = "#4ecb8d"
RED        = "#e05a5a"
ORANGE     = "#e8923a"

PAD = 20   # consistent horizontal padding

# ── Network ───────────────────────────────────────────────────────────────────
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()

def scan_for_webos(timeout=1.0):
    local_ip = get_local_ip()
    subnet   = ".".join(local_ip.split(".")[:3])

    def check(ip):
        try:
            s = socket.socket()
            s.settimeout(timeout)
            s.connect((ip, 3000))
            s.close()
            return ip
        except:
            return None

    ips = [f"{subnet}.{i}" for i in range(1, 255)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=80) as ex:
        results = ex.map(check, ips)
    return [ip for ip in results if ip]

def fetch_apps(ip):
    result = subprocess.run(
        ["python", "main.py", "--webos-host", ip, "--list-apps"],
        capture_output=True, text=True, timeout=20
    )
    app_dict = {}
    for line in result.stdout.strip().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            app_dict[parts[1].strip()] = parts[0].strip()
    return dict(sorted(app_dict.items()))


# ── App ───────────────────────────────────────────────────────────────────────
class App:
    def __init__(self, root):
        self.root    = root
        self.options = {}

        root.title("Gesture Remote")
        root.configure(bg=BG)
        root.resizable(True, True)

        self._build()

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build(self):
        r = self.root

        # Header
        tk.Label(r, text="GESTURE REMOTE", bg=BG, fg=ACCENT,
                 font=("Courier New", 18, "bold"),
                 pady=20).pack(fill=tk.X)

        tk.Label(r, text="webOS TV Controller", bg=BG, fg=TEXT_DIM,
                 font=("Courier New", 9)).pack()

        self._divider()

        # Connection section
        tk.Label(r, text="TV CONNECTION", bg=BG, fg=TEXT_DIM,
                 font=("Courier New", 8),
                 anchor="w").pack(fill=tk.X, padx=PAD, pady=(12, 4))

        self.ip_var = tk.StringVar()
        ip_entry = tk.Entry(r,
            textvariable=self.ip_var,
            bg=BG3, fg=TEXT, insertbackground=ACCENT,
            relief=tk.FLAT, highlightthickness=1,
            highlightbackground=BORDER, highlightcolor=ACCENT,
            font=("Courier New", 12))
        ip_entry.pack(fill=tk.X, padx=PAD, ipady=10)

        self._btn(r, "SCAN NETWORK FOR TV",
                  self._discover, accent=False).pack(
            fill=tk.X, padx=PAD, pady=(8, 0), ipady=8)

        self._btn(r, "CONNECT  &  LOAD APPS",
                  self._connect, accent=True).pack(
            fill=tk.X, padx=PAD, pady=(6, 0), ipady=8)

        self.status_var = tk.StringVar(value="Enter IP or scan, then connect")
        self.status_lbl = tk.Label(r, textvariable=self.status_var,
                                   bg=BG, fg=TEXT_DIM,
                                   font=("Courier New", 9),
                                   wraplength=400, justify="left", anchor="w")
        self.status_lbl.pack(fill=tk.X, padx=PAD, pady=(6, 8))

        self._divider()

        # Presets section
        tk.Label(r, text="APP PRESETS", bg=BG, fg=TEXT_DIM,
                 font=("Courier New", 8),
                 anchor="w").pack(fill=tk.X, padx=PAD, pady=(12, 2))

        tk.Label(r, text="Assign apps to left-hand finger count gestures",
                 bg=BG, fg=TEXT_DIM, font=("Courier New", 8),
                 anchor="w").pack(fill=tk.X, padx=PAD, pady=(0, 8))

        self.preset_vars = []
        labels = ["1 finger", "2 fingers", "3 fingers", "4 fingers", "5 fingers"]
        self.menus = []

        for label in labels:
            var = tk.StringVar(value="— not set —")
            self.preset_vars.append(var)

            row = tk.Frame(r, bg=BG2, highlightthickness=1,
                           highlightbackground=BORDER)
            row.pack(fill=tk.X, padx=PAD, pady=2)

            tk.Label(row, text=label, bg=BG2, fg=TEXT_DIM,
                     font=("Courier New", 9), width=10,
                     anchor="w").pack(side=tk.LEFT, padx=(10, 0), pady=8)

            om = tk.OptionMenu(row, var, "— not set —")
            om.config(
                bg=BG3, fg=TEXT,
                activebackground=ACCENT, activeforeground=BG,
                relief=tk.FLAT, highlightthickness=0,
                font=("Courier New", 9), anchor="w",
                width=28
            )
            om["menu"].config(
                bg=BG3, fg=TEXT,
                activebackground=ACCENT, activeforeground=BG,
                font=("Courier New", 9), relief=tk.FLAT
            )
            om.pack(side=tk.LEFT, fill=tk.X, expand=True,
                    padx=(4, 8), pady=6)
            self.menus.append((var, om))

        self._divider()

        # Launch
        self.launch_btn = self._btn(r, "LAUNCH GESTURE REMOTE",
                                    self._launch, accent=True)
        self.launch_btn.pack(fill=tk.X, padx=PAD, pady=16, ipady=10)
        self.launch_btn.config(state=tk.DISABLED,
                               bg=BG2, fg=TEXT_DIM,
                               activebackground=BG2)

        self._scan_btn  = None   # stored reference if needed
        self._conn_btn  = None

    def _divider(self):
        tk.Frame(self.root, bg=BORDER, height=1).pack(
            fill=tk.X, padx=PAD, pady=4)

    def _btn(self, parent, text, cmd, accent=False):
        c = ACCENT if accent else BG3
        fg = BG if accent else TEXT_DIM
        b = tk.Button(parent, text=text, command=cmd,
                      bg=c, fg=fg,
                      activebackground=ACCENT_DIM, activeforeground=BG,
                      relief=tk.FLAT, font=("Courier New", 10, "bold"),
                      cursor="hand2", bd=0)
        return b

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _status(self, msg, color=TEXT_DIM):
        self.status_var.set(msg)
        self.status_lbl.config(fg=color)
        self.root.update_idletasks()

    def _repopulate(self, options):
        for var, om in self.menus:
            menu = om["menu"]
            menu.delete(0, "end")
            menu.add_command(label="— not set —",
                             command=lambda v=var: v.set("— not set —"))
            for name in sorted(options.keys()):
                menu.add_command(label=name,
                                 command=lambda n=name, v=var: v.set(n))
            var.set("— not set —")

    # ── Actions ───────────────────────────────────────────────────────────────
    def _discover(self):
        self._status("Scanning network...", ORANGE)

        def run():
            found = scan_for_webos(timeout=1.0)
            if found:
                self.ip_var.set(found[0])
                self._status(f"Found webOS TV at {found[0]} — click Connect", GREEN)
            else:
                self._status("No webOS TV found. Enter IP manually.", RED)

        threading.Thread(target=run, daemon=True).start()

    def _connect(self):
        ip = self.ip_var.get().strip()
        if not ip:
            self._status("Enter a TV IP address first.", RED)
            return

        self._status(f"Connecting to {ip}...", ORANGE)

        def run():
            try:
                apps = fetch_apps(ip)
                if not apps:
                    self._status("Connected but no apps returned.", RED)
                    return
                self.options = apps
                self._repopulate(apps)
                self._status(f"Connected  ·  {len(apps)} apps loaded", GREEN)
                self.launch_btn.config(
                    state=tk.NORMAL, bg=ACCENT, fg=BG)
            except subprocess.TimeoutExpired:
                self._status("Timed out. Is the TV on?", RED)
            except Exception as e:
                self._status(f"Error: {e}", RED)

        threading.Thread(target=run, daemon=True).start()

    def _launch(self):
        ip = self.ip_var.get().strip()
        if not ip:
            self._status("No IP set.", RED)
            return

        presets = []
        for i, (var, _) in enumerate(self.menus):
            name = var.get()
            if name != "— not set —" and name in self.options:
                presets.append(f"{i+1}:{self.options[name]}")

        cmd = ["python", "main.py", "--webos-host", ip]
        if presets:
            cmd += ["--presets", ",".join(presets)]

        print(f"Launching: {' '.join(cmd)}")
        subprocess.Popen(cmd)
        self._status("Gesture remote launched ✓", GREEN)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()