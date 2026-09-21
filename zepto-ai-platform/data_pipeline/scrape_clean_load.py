"""
Zepto Capstone - Module 1: Data Pipeline
=========================================
Scrapes books.toscrape.com, cleans the fields, converts GBP -> INR using a
fixed baseline rate, loads everything into a normalized SQLite database,
and runs the required SQL + pandas queries.

Run:
    python scrape_clean_load.py

Requires internet access to books.toscrape.com (no login / no API key needed).
"""

import re
import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
CATALOGUE_URL = BASE_URL + "catalogue/"
DB_PATH = Path(__file__).parent / "books.db"

# Project-defined fixed baseline conversion rate (NOT a live/historical rate).
GBP_TO_INR = 105.50

RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


# ---------------------------------------------------------------------------
# 1. SCRAPE
# ---------------------------------------------------------------------------
def get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def scrape_category_listing(category_url: str, category_name: str) -> list[dict]:
    """Scrape every book across all paginated pages of one category listing."""
    books = []
    url = category_url
    while url:
        soup = get_soup(url)
        for article in soup.select("article.product_pod"):
            title = article.h3.a["title"].strip()
            price_text = article.select_one("p.price_color").get_text(strip=True)
            rating_word = article.select_one("p.star-rating")["class"][1]
            availability = article.select_one("p.instock.availability").get_text(strip=True)
            books.append(
                {
                    "title": title,
                    "price_text": price_text,
                    "rating_text": rating_word,
                    "availability_text": availability,
                    "category": category_name,
                }
            )
        next_link = soup.select_one("li.next a")
        url = (url.rsplit("/", 1)[0] + "/" + next_link["href"]) if next_link else None
        time.sleep(0.2)  # be polite to the practice site
    return books


def discover_categories(limit: int = 5) -> list[tuple[str, str]]:
    """Return (name, url) for the first `limit` real categories on the site."""
    soup = get_soup(BASE_URL)
    cats = []
    for a in soup.select("div.side_categories ul li ul li a"):
        name = a.get_text(strip=True)
        href = a["href"]  # e.g. catalogue/category/books/travel_2/index.html
        url = BASE_URL + href
        cats.append((name, url))
    return cats[:limit]


def scrape_all(min_rows: int = 60) -> pd.DataFrame:
    """Scrape at least 3 categories (more if needed) until we clear min_rows."""
    categories = discover_categories(limit=5)
    all_rows: list[dict] = []
    for name, url in categories:
        rows = scrape_category_listing(url, name)
        all_rows.extend(rows)
        print(f"  scraped {len(rows):3d} books from category '{name}'")
        if len(all_rows) >= min_rows and len(categories[: categories.index((name, url)) + 1]) >= 3:
            break
    df = pd.DataFrame(all_rows)
    print(f"Total scraped rows: {len(df)} across "
          f"{df['category'].nunique()} categories")
    return df


# ---------------------------------------------------------------------------
# 2. CLEAN
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # price_gbp: strip currency symbol -> float
    df["price_gbp"] = (
        df["price_text"].str.replace(r"[^\d.]", "", regex=True).astype(float)
    )

    # rating: word -> int, drop rows that fail to parse (design choice, see README)
    df["rating"] = df["rating_text"].map(RATING_WORDS)
    before = len(df)
    df = df.dropna(subset=["rating"])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} row(s) with unparseable rating text.")
    df["rating"] = df["rating"].astype(int)

    # in_stock: availability text -> bool
    df["in_stock"] = df["availability_text"].str.contains("In stock", case=False, na=False)

    # Any remaining NaNs in the numeric price column: median-impute rather than drop,
    # since price is usually recoverable/central-tendency-safe while rating/availability
    # parsing failures indicate a genuinely malformed row.
    if df["price_gbp"].isna().any():
        median_price = df["price_gbp"].median()
        n_missing = df["price_gbp"].isna().sum()
        df["price_gbp"] = df["price_gbp"].fillna(median_price)
        print(f"Median-imputed {n_missing} missing price_gbp value(s) with {median_price:.2f}")

    # price_inr: fixed baseline conversion, project-defined constant (no date reference)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    return df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]


# ---------------------------------------------------------------------------
# 3. LOAD into normalized SQLite schema
# ---------------------------------------------------------------------------
def build_database(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            category_name TEXT UNIQUE
        );

        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY,
            title TEXT,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock INTEGER,
            category_id INTEGER REFERENCES categories(category_id)
        );
        """
    )

    categories = sorted(df["category"].unique())
    cat_id_map = {}
    for name in categories:
        cur.execute("INSERT INTO categories (category_name) VALUES (?)", (name,))
        cat_id_map[name] = cur.lastrowid

    for _, row in df.iterrows():
        cur.execute(
            """INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                row["title"],
                row["price_gbp"],
                row["price_inr"],
                int(row["rating"]),
                int(bool(row["in_stock"])),
                cat_id_map[row["category"]],
            ),
        )
    conn.commit()
    conn.close()
    print(f"Loaded {len(df)} books into {db_path}")


# ---------------------------------------------------------------------------
# 4. QUERY - at least 5 SQL queries covering the required clauses
# ---------------------------------------------------------------------------
QUERIES = {
    "q1_select_where": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE in_stock = 1
    """,
    "q2_order_by": """
        SELECT title, price_inr
        FROM books
        ORDER BY price_inr DESC
    """,
    "q3_limit": """
        SELECT title, rating
        FROM books
        ORDER BY rating DESC
        LIMIT 10
    """,
    "q4_distinct": """
        SELECT DISTINCT category_name
        FROM categories
    """,
    "q5_between_in": """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 10 AND 30
          AND rating IN (4, 5)
    """,
    "q6_join_top_rated_per_category": """
        SELECT c.category_name, b.title, b.rating, b.price_inr
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE b.rating = (
            SELECT MAX(b2.rating) FROM books b2 WHERE b2.category_id = b.category_id
        )
        ORDER BY c.category_name
        LIMIT 10
    """,
}


def run_queries(db_path: Path = DB_PATH) -> dict[str, pd.DataFrame]:
    conn = sqlite3.connect(db_path)
    results = {}
    for name, sql in QUERIES.items():
        df = pd.read_sql(sql, conn)
        results[name] = df
        print(f"\n--- {name} ---\n{df.head(10)}")
    conn.close()
    return results


def verify_join_with_pandas(db_path: Path = DB_PATH) -> None:
    """Reproduce the JOIN query result purely with pandas.merge (no SQL join)."""
    conn = sqlite3.connect(db_path)
    books = pd.read_sql("SELECT * FROM books", conn)
    categories = pd.read_sql("SELECT * FROM categories", conn)
    conn.close()

    merged = books.merge(categories, on="category_id")
    idx = merged.groupby("category_id")["rating"].transform("max") == merged["rating"]
    top_rated = (
        merged.loc[idx, ["category_name", "title", "rating", "price_inr"]]
        .sort_values("category_name")
        .head(10)
    )
    print("\n--- pandas.merge equivalent of q6_join ---")
    print(top_rated)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Scraping books.toscrape.com ...")
    raw_df = scrape_all(min_rows=60)

    print("\nCleaning data ...")
    clean_df = clean(raw_df)
    clean_df.to_csv(Path(__file__).parent / "books_clean.csv", index=False)

    print("\nBuilding SQLite database ...")
    build_database(clean_df)

    print("\nRunning SQL queries ...")
    run_queries()

    print("\nVerifying JOIN query with pandas.merge ...")
    verify_join_with_pandas()

    print("\nDone.")
