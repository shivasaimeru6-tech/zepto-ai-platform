# Module 1 — Data Pipeline

## Run
```bash
pip install -r requirements.txt   # or the root requirements.txt
python scrape_clean_load.py
```
This scrapes books.toscrape.com, writes `books_clean.csv`, builds `books.db`,
prints 5 SQL query results, and prints the pandas.merge equivalent of the
JOIN query for comparison.

## Design decisions
- **Categories/pages scraped**: the first 5 real categories listed on the
  site's sidebar, paginated fully within each, stopping once we have ≥60
  rows across ≥3 categories.
- **Currency conversion**: fixed baseline rate **1 GBP = 105.50 INR**, a
  project-defined constant (not a live/historical rate), applied directly to
  `price_gbp` to produce `price_inr`.
- **Row handling on parse failure**:
  - `rating` (word → int): rows where the rating text doesn't match one of
    One..Five are **dropped** — an unparseable rating means we can't trust
    the row's core signal, and this text is a closed vocabulary on the site
    so failures are true anomalies, not just noisy numeric data.
  - `price_gbp`: any residual missing values are **median-imputed** rather
    than dropped, since price is a continuous field where the median is a
    defensible central-tendency fallback and dropping would lose otherwise
    valid rows for a single hiccuped field.
- **Schema**: `categories(category_id PK, category_name)` and
  `books(book_id PK, title, price_gbp, price_inr, rating, in_stock,
  category_id FK)`.

## Files
- `scrape_clean_load.py` — scrape → clean → convert → load → query, all in one script.
- `books_clean.csv` — cleaned dataset (generated on run).
- `books.db` — SQLite database (generated on run; regenerate anytime via the script).
