import fitz

with fitz.open("data/useful/Adams_1989.pdf") as doc:
    for page_num, page in enumerate(doc, start=1):
        tabs = page.find_tables()
        print(f"Page {page_num}: {len(tabs.tables)} tables found")
        if tabs.tables:
            for i, tab in enumerate(tabs.tables):
                print(f"  Table {i + 1}: {tab.bbox}")
