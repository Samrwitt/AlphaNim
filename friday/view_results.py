"""Open the real script output in a simple window for inspection/screenshots."""
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

SCRIPT = Path(__file__).with_name('random_nim.py')


def run(*arguments):
    command = [sys.executable, '-B', str(SCRIPT), *arguments]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


root = tk.Tk()
root.title('Friday Nim - actual Python output')
root.geometry('1180x920')
tabs = ttk.Notebook(root)
tabs.pack(fill='both', expand=True)
generation = run()
for label, output in [('Complete games', generation + '\n' + run('--show', 'games')),
                      ('Saved CSV rows', run('--show', 'rows'))]:
    frame = ttk.Frame(tabs)
    tabs.add(frame, text=label)
    text = tk.Text(frame, font=('DejaVu Sans Mono', 12), wrap='none', padx=18, pady=14)
    text.pack(fill='both', expand=True)
    text.insert('1.0', f'Actual stdout: python3 friday/random_nim.py'
                f'{" --show rows" if label == "Saved CSV rows" else " / --show games"}\n\n' + output)
    text.configure(state='disabled')
if '--rows' in sys.argv:
    tabs.select(1)
if '--capture' in sys.argv:
    # Native desktop capture of this running window, not a rendered mock image.
    destination = sys.argv[sys.argv.index('--capture') + 1]
    def capture():
        root.update_idletasks()
        subprocess.run(['import', '-window', str(root.winfo_id()), destination], check=True)
        root.destroy()
    root.after(1000, capture)
root.mainloop()
