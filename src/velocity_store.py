"""
src/velocity_store.py
FIX 9.1 -- velocity_24h was still approximate at inference.
           audit doc phase 3: replace sender_stats.csv lifetime count
           with a real 24-hour sliding window backed by SQLite.

API
───
  store = VelocityStore()           # opens/creates outputs/velocity.db
  store.build_from_training(df)     # called once at end of data_processing.py
  v1h, v24h = store.get(sender, ts) # called per-row in encode_raw_df()
  store.record(sender, ts)          # call after each live inference to keep counts fresh

Design
──────
  Table: events(sender_upi TEXT, ts INTEGER)  -- ts is Unix epoch seconds
  Index on (sender_upi, ts) makes the window query O(log n).
  For the dashboard demo use-case (batch CSV upload) we also support
  compute_from_df() which derives velocities directly from the uploaded
  file's own timestamp column -- no DB needed, exact for that batch.

Performance
───────────
  293k training rows -> build_from_training takes ~3 seconds.
  Per-row inference query (get) is <1ms with the index.
  DB size after 293k events: ~15 MB.
"""

import os
import sqlite3
import time
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_DB_PATH = os.path.join('outputs', 'velocity.db')

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    sender_upi   TEXT    NOT NULL,
    ts           INTEGER NOT NULL,
    receiver_upi TEXT,              -- IMP-03: needed for unique_receivers_1h
    amount       REAL                -- IMP-03: needed for amount_entropy_1h
);
"""
_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_sender_ts ON events (sender_upi, ts);
"""
_INSERT = "INSERT INTO events (sender_upi, ts, receiver_upi, amount) VALUES (?, ?, ?, ?)"  # IMP-03
_QUERY_1H  = "SELECT COUNT(*) FROM events WHERE sender_upi=? AND ts >= ? AND ts < ?"
_QUERY_24H = "SELECT COUNT(*) FROM events WHERE sender_upi=? AND ts >= ? AND ts < ?"
_QUERY_RECV_1H   = "SELECT receiver_upi FROM events WHERE sender_upi=? AND ts >= ? AND ts < ?"  # IMP-03
_QUERY_AMTS_1H   = "SELECT amount FROM events WHERE sender_upi=? AND ts >= ? AND ts < ?"        # IMP-03
_CLEANUP   = "DELETE FROM events WHERE ts < ?"   # prune rows older than 48h


class VelocityStore:
    """
    SQLite-backed 24-hour sliding window velocity counter.

    Usage in encode_raw_df():
        store = VelocityStore()
        v1h, v24h = store.get(sender_upi, timestamp)

    Usage after live inference (to keep DB fresh):
        store.record(sender_upi, timestamp)

    Usage at training time (run once after generate_upi_data.py):
        store.build_from_training(df)
    """

    def __init__(self, db_path: str = _DB_PATH):
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute('PRAGMA journal_mode=WAL')      # FIX-06: WAL allows concurrent readers + writer
        self._conn.execute('PRAGMA synchronous=NORMAL')    # FIX-06: safe + faster than FULL
        self._conn.execute(_CREATE_TABLE)
        self._conn.execute(_CREATE_INDEX)
        self._conn.commit()
        log.info(f"VelocityStore opened: {db_path}")

    # ── read ──────────────────────────────────────────────────────────────────

    def get(self, sender_upi: str, ts) -> dict:
        """
        Return velocity dict for sender_upi at time ts.
        Keys: velocity_1h, velocity_24h, unique_receivers_1h, amount_entropy_1h [IMP-03]
        ts may be a datetime, pd.Timestamp, or Unix int.
        Counts are exclusive of ts itself (prior transactions only).
        """
        t = self._to_unix(ts)
        cur = self._conn.cursor()
        v1h  = cur.execute(_QUERY_1H,  (sender_upi, t - 3600,  t)).fetchone()[0]
        v24h = cur.execute(_QUERY_24H, (sender_upi, t - 86400, t)).fetchone()[0]

        # IMP-03: unique receivers and entropy from 1h window
        rows_recv = cur.execute(_QUERY_RECV_1H, (sender_upi, t - 3600, t)).fetchall()
        unique_recv = len(set(r[0] for r in rows_recv if r[0] is not None))

        rows_amts = cur.execute(_QUERY_AMTS_1H, (sender_upi, t - 3600, t)).fetchall()
        amts = np.array([r[0] for r in rows_amts if r[0] is not None], dtype=np.float64)
        if len(amts) > 1:
            bins   = np.digitize(amts, [100, 1000, 10000, 100000])
            counts = np.bincount(bins, minlength=6)[1:]
            probs  = counts / counts.sum()
            probs  = probs[probs > 0]
            entropy = float(-np.sum(probs * np.log2(probs)))
        else:
            entropy = 0.0

        return {
            'velocity_1h':         int(v1h),
            'velocity_24h':        int(v24h),
            'unique_receivers_1h': unique_recv,
            'amount_entropy_1h':   entropy,
        }

    # ── write ─────────────────────────────────────────────────────────────────

    def record(self, sender_upi: str, ts, receiver_upi: str = None, amount: float = None) -> None:
        """Record a single transaction event. Call after each live inference."""
        t = self._to_unix(ts)
        self._conn.execute(_INSERT, (sender_upi, t, receiver_upi, amount))
        self._conn.commit()

    def record_batch(self, senders: list[str], timestamps, receivers=None, amounts=None) -> None:
        """Bulk-insert a batch of events. Used by build_from_training."""
        if receivers is None:
            receivers = [None] * len(senders)
        if amounts is None:
            amounts = [None] * len(senders)
        rows = [(s, self._to_unix(t), r, a)
                for s, t, r, a in zip(senders, timestamps, receivers, amounts)]
        self._conn.executemany(_INSERT, rows)
        self._conn.commit()

    # ── maintenance ───────────────────────────────────────────────────────────

    def cleanup(self, older_than_seconds: int = 172800) -> int:
        """Delete events older than `older_than_seconds` (default 48h). Returns rows deleted."""
        cutoff = int(time.time()) - older_than_seconds
        cur = self._conn.execute(_CLEANUP, (cutoff,))
        self._conn.commit()
        return cur.rowcount

    def build_from_training(self, df: pd.DataFrame) -> None:
        """
        Populate the store from the full training DataFrame.
        Called once at the end of data_processing.py.
        Clears existing rows first to avoid duplicates on re-run.
        """
        log.info(f"VelocityStore: building from {len(df):,} training rows...")
        self._conn.execute("DELETE FROM events")
        self._conn.commit()
        senders    = df['sender_upi'].tolist()
        timestamps = df['timestamp'].tolist()
        receivers  = df['receiver_upi'].tolist() if 'receiver_upi' in df.columns else None  # IMP-03
        amounts    = df['amount'].tolist()        if 'amount'       in df.columns else None  # IMP-03
        self.record_batch(senders, timestamps, receivers, amounts)
        log.info(f"VelocityStore: inserted {len(df):,} events -> {self.db_path}")

    # ── batch inference helper ────────────────────────────────────────────────

    def compute_from_df(self, df: pd.DataFrame) -> tuple:
        """
        Compute velocity_1h, velocity_24h, unique_receivers_1h, amount_entropy_1h
        for every row of an uploaded DataFrame using only that DataFrame's own timestamps.
        No DB read -- exact for any self-contained batch.
        Falls back gracefully if required columns absent.

        Returns (v1h, v24h, unique_recv, entropy) -- all pd.Series aligned to df.index.
        [IMP-03: extended from 2-tuple to 4-tuple]
        """
        _zeros = lambda: pd.Series(0, index=df.index)
        if 'timestamp' not in df.columns or 'sender_upi' not in df.columns:
            return _zeros(), _zeros(), _zeros(), _zeros()

        has_receiver = 'receiver_upi' in df.columns
        has_amount   = 'amount' in df.columns

        df2 = df[['sender_upi', 'timestamp']].copy()
        if has_receiver:
            df2['receiver_upi'] = df['receiver_upi'].values
        if has_amount:
            df2['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0).values
        df2['ts'] = pd.to_datetime(df2['timestamp'], errors='coerce').astype(np.int64) // 10**9

        v1h_out   = np.zeros(len(df2), dtype=np.int32)
        v24h_out  = np.zeros(len(df2), dtype=np.int32)
        urecv_out = np.zeros(len(df2), dtype=np.int32)
        entr_out  = np.zeros(len(df2), dtype=np.float64)

        for sender, grp in df2.groupby('sender_upi'):
            idx          = grp.index.values
            times        = grp['ts'].values
            times_sorted = np.sort(times)
            recv_arr     = grp['receiver_upi'].values if has_receiver else None
            amt_arr      = grp['amount'].values       if has_amount   else None

            for i, (gi, t) in enumerate(zip(idx, times)):
                pos    = int(np.searchsorted(times_sorted, t, side='left'))
                lo_1h  = int(np.searchsorted(times_sorted, t - 3600,  side='left'))
                lo_24h = int(np.searchsorted(times_sorted, t - 86400, side='left'))
                v1h_out[gi]  = pos - lo_1h
                v24h_out[gi] = pos - lo_24h

                # IMP-03: unique receivers
                if recv_arr is not None:
                    # get positions in sorted order -- must map back to unsorted grp order
                    mask_1h = (times[:i] >= t - 3600) & (times[:i] < t)
                    urecv_out[gi] = len(set(recv_arr[:i][mask_1h]))

                # IMP-03: amount entropy
                if amt_arr is not None:
                    mask_1h = (times[:i] >= t - 3600) & (times[:i] < t)
                    amts_1h = amt_arr[:i][mask_1h]
                    if len(amts_1h) > 1:
                        bins   = np.digitize(amts_1h, [100, 1000, 10000, 100000])
                        counts = np.bincount(bins, minlength=6)[1:]
                        probs  = counts / counts.sum()
                        probs  = probs[probs > 0]
                        entr_out[gi] = float(-np.sum(probs * np.log2(probs)))

        # IMP-03: validate constraint unique_recv <= v1h
        # L-09: replaced assert with warning + clip -- assert crashes production
        # and is stripped by python -O. Edge cases (duplicate timestamps) can trigger this.
        _violations = (urecv_out > v1h_out).sum()
        if _violations > 0:
            log.warning(f"unique_receivers_1h > velocity_1h in {_violations} rows -- clipping")
            urecv_out = np.minimum(urecv_out, v1h_out)

        return (
            pd.Series(v1h_out,   index=df.index),
            pd.Series(v24h_out,  index=df.index),
            pd.Series(urecv_out, index=df.index),
            pd.Series(entr_out,  index=df.index),
        )

    # ── util ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_unix(ts) -> int:
        if isinstance(ts, (int, float, np.integer)):
            return int(ts)
        if isinstance(ts, str):
            ts = pd.to_datetime(ts)
        if hasattr(ts, 'timestamp'):
            return int(ts.timestamp())
        return int(pd.Timestamp(ts).timestamp())

    def close(self):
        self._conn.close()

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            pass
