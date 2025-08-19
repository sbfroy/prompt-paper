from pypdf import PdfReader

"""
Raw data extraction from various sources
"""

def read_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def read_website(url):
    pass

def read_wikipedia(topic):
    pass