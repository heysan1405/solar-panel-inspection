
from pathlib import Path
import ultralytics

package = Path(ultralytics.__file__).parent
config = package / "cfg" / "models" / "26" / "yolo26.yaml"

print("Reading:", config)

text = config.read_text(encoding="utf-8")
print(text)
