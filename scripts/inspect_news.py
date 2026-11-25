import os, duckdb
from rltrading.config import RAW_DIR
con = duckdb.connect(); con.execute("SET memory_limit='3GB'")
base = RAW_DIR/"news"/"data"
for name in sorted(os.listdir(base)):
    g = f"'{base}/{name}/*.parquet'"
    r = con.execute(f"select count(*), min(date), max(date), median(length(text)) from read_parquet({g})").fetchone()
    yrs = con.execute(f"select substr(date,1,4) y, count(*) from read_parquet({g}) group by 1 order by 1").fetchall()
    s = con.execute(f"select text, extra_fields from read_parquet({g}) using sample 1 rows").fetchone()
    print(f"== {name}: n={r[0]:,} {r[1]}..{r[2]} medlen={r[3]}")
    print("   years:", " ".join(f"{y}:{c}" for y,c in yrs))
    print("   extra:", s[1][:300])
    print("   text:", s[0][:150].replace("\n"," | "), flush=True)
