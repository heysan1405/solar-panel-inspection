
import inspect
from ultralytics.nn import tasks

source = inspect.getsource(tasks.parse_model)
lines = source.splitlines()

for i, line in enumerate(lines):
    if "base_modules" in line or "repeat_modules" in line:
        start = max(0, i - 2)
        end = min(len(lines), i + 12)
        print(f"\n--- Lines around {i + 1} ---")
        print("\n".join(
            f"{j + 1}: {lines[j]}"
            for j in range(start, end)
        ))
