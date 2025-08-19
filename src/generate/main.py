from src.config import CONFIG
from src.scope.readers import from_pdf

print(from_pdf(CONFIG["scope"]["pdfs"][0]))
