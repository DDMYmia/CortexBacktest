"""SQL over the hive-partitioned Parquet research layer.

The research layer is 118 GB of minute bars in `year=YYYY/part-YYYY-MM.parquet`.
Until now the only way to read it was a Python process on the machine holding
it, which meant every question -- "how many names traded in 2021", "what were
the twenty largest by dollar volume last June" -- required writing a scan.
DuckDB answers those in SQL against the files in place, with no load step and no
copy.

It is worth the dependency because the pruning is real, measured against the
live 2.67-billion-row layer:

    one symbol, one year                    0.15 s
    one symbol, thirty-two years            3.87 s
    a year of one symbol aggregated to days 1.26 s
    a month's cross-section, ranked         0.06 s

## The timezone trap, which this module exists to close

Casting a `TIMESTAMP WITH TIME ZONE` to `DATE` in DuckDB uses the *session*
timezone, and the session default is whatever timezone the machine is set to.
So `timestamp::DATE` does not mean "the UTC date" and does not mean "the trading
day" -- it means "the date in the timezone of whoever happened to run the query".
The same aggregate gives different sessions on a laptop in Auckland and a server
in UTC, with nothing in the result saying so.

Two things follow, and this module does both.

`connect` pins `TimeZone='UTC'`, so a raw `timestamp::DATE` in the escape hatch
is at least the same everywhere rather than machine-dependent.

Every helper then converts to the *exchange* timezone before taking a date,
because UTC is reproducible and still wrong: a US session runs 09:30-16:00 New
York and extended hours run to 20:00, which is 00:00-01:00 the next UTC day, so
one session lands on two dates. Counting AAPL's distinct dates in 2023 on the
live layer:

    machine-local (Pacific/Auckland)  303
    UTC                               268
    America/New_York                  250

NYSE held exactly 250 sessions in 2023. Only the third is the trading day, and
the first is what an unqualified `::DATE` returns on this hardware.

What this does not do is know about half days. 2023-11-24 comes back with 402
bars against a normal session's ~960, which is correct and visible, but the
1pm close itself is only knowable from an exchange calendar --
`TradingCalendar.from_exchange` -- because trading continues past it in the data.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb
    import polars as pl

# The exchange these bars come from. Held as a constant rather than defaulted at
# each call site so that a study cannot silently mix two conventions.
DEFAULT_EXCHANGE_TZ = "America/New_York"

_MISSING_DUCKDB = (
    "DuckDB is not installed. Install cortexbacktest[duck] to query the Parquet "
    "research layer in SQL."
)


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


@dataclass(frozen=True, slots=True)
class ResearchStore:
    """A DuckDB session over a hive-partitioned bar layer.

    `root` is the directory holding `year=YYYY/` partitions. Nothing is copied
    or loaded: the files are read where they are, and a query that names a year
    or a ticker reads only the partitions and row groups it needs.
    """

    root: str
    exchange_tz: str = DEFAULT_EXCHANGE_TZ
    threads: int = 4
    memory_limit: str = "4GB"

    @property
    def glob(self) -> str:
        return f"{self.root.rstrip('/')}/year=*/*.parquet"

    def source(self) -> str:
        """The FROM clause for the whole layer, with hive columns exposed."""
        return f"read_parquet({_sql_string(self.glob)}, hive_partitioning=true)"

    @contextmanager
    def connect(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """Open a connection with `bars` registered as a view.

        Threads and memory are capped rather than left to DuckDB's defaults,
        which size themselves to the machine. This layer is usually queried on
        the data server while other jobs are running there, and an unbounded
        aggregate over 2.67 billion rows will take everything it is given.
        """
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(_MISSING_DUCKDB) from exc

        connection = duckdb.connect()
        try:
            connection.execute(f"SET threads={int(self.threads)}")
            connection.execute(f"SET memory_limit={_sql_string(self.memory_limit)}")
            # Pinned so that a bare `timestamp::DATE` means the same thing on
            # every machine. DuckDB otherwise defaults the session timezone to
            # the host's, silently making date bucketing depend on where the
            # query ran. The helpers below still convert to the exchange zone;
            # this only removes the machine from the equation.
            connection.execute("SET TimeZone='UTC'")
            connection.execute(f"CREATE VIEW bars AS SELECT * FROM {self.source()}")
            yield connection
        finally:
            connection.close()

    def sql(self, query: str, params: Sequence[Any] | None = None) -> list[tuple[Any, ...]]:
        """Run one query against `bars` and return its rows.

        The escape hatch: anything the helpers below do not cover is a plain
        SELECT away, and the view is already registered.
        """
        with self.connect() as connection:
            return connection.execute(query, params or []).fetchall()

    def frame(self, query: str, params: Sequence[Any] | None = None) -> pl.DataFrame:
        """Same, as a Polars frame, which is what the rest of this package speaks."""
        with self.connect() as connection:
            return connection.execute(query, params or []).pl()

    # --- session-aware helpers -------------------------------------------

    def _session_date(self, column: str = "timestamp") -> str:
        return f"({column} AT TIME ZONE {_sql_string(self.exchange_tz)})::DATE"

    def daily_bars(
        self,
        symbol: str,
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> pl.DataFrame:
        """Minute bars aggregated to one row per trading session.

        Grouped on the exchange-local date, so a session and its extended hours
        stay together. Open and close are taken by timestamp order rather than
        by `min`/`max`, which would return the lowest and highest prices of the
        day and call them the open and the close.
        """
        where = ["ticker = ?"]
        params: list[Any] = [symbol]
        if start:
            where.append(f"{self._session_date()} >= ?::DATE")
            params.append(start)
        if end:
            where.append(f"{self._session_date()} <= ?::DATE")
            params.append(end)

        return self.frame(
            f"""
            SELECT {self._session_date()} AS session_date,
                   first(open  ORDER BY timestamp) AS open,
                   max(high)                       AS high,
                   min(low)                        AS low,
                   last(close ORDER BY timestamp)  AS close,
                   sum(volume)                     AS volume,
                   count(*)                        AS bar_count
            FROM bars
            WHERE {' AND '.join(where)}
            GROUP BY session_date
            ORDER BY session_date
            """,
            params,
        )

    def universe_by_year(self) -> pl.DataFrame:
        """Distinct tickers per year.

        The check that catches a universe discontinuity: a year where the count
        jumps by thousands is a change in what the vendor included, not a change
        in the market, and any cross-sectional study spanning it is comparing
        two different populations.
        """
        return self.frame(
            "SELECT year, count(DISTINCT ticker) AS symbols FROM bars GROUP BY year ORDER BY year"
        )

    def liquidity_ranking(
        self,
        start: str,
        end: str,
        *,
        limit: int = 100,
    ) -> pl.DataFrame:
        """Tickers ranked by traded value over a window, plus their coverage.

        `sessions` is returned alongside because a name can rank highly on a
        handful of frantic days and be untradeable the rest of the window.
        Ranking on value alone hides that; this makes it one column away.
        """
        return self.frame(
            f"""
            SELECT ticker,
                   sum(close * volume)                    AS dollar_volume,
                   count(DISTINCT {self._session_date()}) AS sessions,
                   min({self._session_date()})            AS first_session,
                   max({self._session_date()})            AS last_session
            FROM bars
            WHERE {self._session_date()} >= ?::DATE AND {self._session_date()} <= ?::DATE
            GROUP BY ticker
            ORDER BY dollar_volume DESC
            LIMIT {int(limit)}
            """,
            [start, end],
        )

    def session_coverage(self, symbol: str, year: int) -> pl.DataFrame:
        """Bars per session for one symbol and year, thinnest first.

        Reading this rather than trusting the layer is how a gap is found. A
        session with a handful of bars is a vendor outage; one with roughly half
        is an early close. Neither is visible in a row count.
        """
        return self.frame(
            f"""
            SELECT {self._session_date()} AS session_date, count(*) AS bars
            FROM bars WHERE ticker = ? AND year = ?
            GROUP BY session_date ORDER BY bars ASC, session_date
            """,
            [symbol, year],
        )
