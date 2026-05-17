"""
generate_upi_data.py
Run ONCE to create data/upi_transactions.csv (~300k rows)

Fraud patterns injected:
  1. Late-night high-value P2P transfers
  2. Rapid-fire velocity (many txns same sender in 1h)
  3. New device + high amount
  4. Cross-state impossible travel
  5. Round-amount transfers
  6. Dormant account suddenly active
  7. Camouflaged high-value (normal-looking, only device+state signal) [IMP-02]
  8. Account takeover simulation (no new device, amount_vs_avg spike)   [IMP-02]

FIXES vs original:
  - Normal transactions now have correlated sender/receiver state:
    80% same state, 20% cross-state. Previously both were random ->
    ~93% of normal txns had cross_state=1 -> feature was pure noise.
    Now cross_state=1 is genuinely a fraud signal.
  - Same fix for cross_bank: normal txns 70% same bank, 30% cross-bank.
  - txn_id uses uuid4 -- original randint caused collisions at 293k rows.
  - velocity_24h computed independently (not velocity_1h × 3).
"""

import os
import sys
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import hashlib
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PATH, RANDOM_SEED

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ─── Constants ────────────────────────────────────────────────────────────────

N_NORMAL  = 290_000
N_FRAUD   = 3_000
N_TOTAL   = N_NORMAL + N_FRAUD

UPI_APPS    = ['GPay', 'PhonePe', 'Paytm', 'BHIM', 'AmazonPay', 'WhatsApp']
APP_WEIGHTS = [0.35,   0.30,      0.20,    0.07,   0.05,         0.03]

BANKS = ['SBI', 'HDFC', 'ICICI', 'Axis', 'Kotak', 'PNB', 'BOB', 'Canara',
         'IndusInd', 'Yes Bank', 'IDFC First', 'UCO Bank']

STATES = [
    'Maharashtra', 'Karnataka', 'Delhi', 'Tamil Nadu', 'Telangana',
    'Gujarat', 'Rajasthan', 'UP', 'West Bengal', 'Kerala',
    'MP', 'Bihar', 'Odisha', 'Punjab', 'Haryana',
]

MERCHANT_CATS = {
    'p2p':        0.38,
    'food':       0.14,
    'utilities':  0.10,
    'fuel':       0.08,
    'groceries':  0.09,
    'rent':       0.05,
    'recharge':   0.07,
    'ecommerce':  0.06,
    'education':  0.03,
}

START_DATE = datetime(2024, 1, 1)
END_DATE   = datetime(2024, 12, 31)
DATE_RANGE = (END_DATE - START_DATE).total_seconds()


def _txn_id():
    # FIX: uuid4 -- no collisions
    return str(uuid.uuid4())


def _random_upi(prefix='user'):
    n    = np.random.randint(1000, 99999)
    bank = random.choice(['okicici', 'oksbi', 'okhdfcbank', 'ybl', 'axl', 'paytm'])
    return f"{prefix}{n}@{bank}"


def _device_hash(uid):
    return hashlib.md5(str(uid).encode()).hexdigest()[:12]


def _rand_ts():
    offset = np.random.uniform(0, DATE_RANGE)
    return START_DATE + timedelta(seconds=offset)


def _amount_normal():
    roll = np.random.random()
    if roll < 0.50:
        return round(np.random.uniform(10, 500), 2)
    elif roll < 0.80:
        return round(np.random.uniform(500, 5000), 2)
    elif roll < 0.95:
        return round(np.random.uniform(5000, 50000), 2)
    else:
        return round(np.random.uniform(50000, 200000), 2)


def _pick_receiver_bank(sender_bank):
    # FIX: normal txns 70% same bank, 30% different -- makes cross_bank a real signal
    if np.random.random() < 0.70:
        return sender_bank
    return random.choice([b for b in BANKS if b != sender_bank])


def _pick_receiver_state(sender_state):
    # FIX: normal txns 80% same state, 20% different -- makes cross_state a real signal
    if np.random.random() < 0.80:
        return sender_state
    return random.choice([s for s in STATES if s != sender_state])


# ─── Generate normal transactions ─────────────────────────────────────────────

print(f"Generating {N_NORMAL:,} normal transactions...")

sender_pool   = [_random_upi('user')  for _ in range(15000)]
receiver_pool = [_random_upi('merch') for _ in range(8000)]
device_pool   = {u: _device_hash(i) for i, u in enumerate(sender_pool)}

# Assign each sender a "home state" -- stable across their normal transactions
sender_home_state = {u: random.choice(STATES) for u in sender_pool}
sender_home_bank  = {u: random.choice(BANKS)  for u in sender_pool}

cats        = list(MERCHANT_CATS.keys())
cat_weights = list(MERCHANT_CATS.values())

normal_rows = []
for _ in range(N_NORMAL):
    sender     = random.choice(sender_pool)
    receiver   = random.choice(receiver_pool)
    ts         = _rand_ts()
    amt        = _amount_normal()
    app        = np.random.choice(UPI_APPS, p=APP_WEIGHTS)
    s_state    = sender_home_state[sender]
    r_state    = _pick_receiver_state(s_state)   # FIX: correlated
    s_bank     = sender_home_bank[sender]
    r_bank     = _pick_receiver_bank(s_bank)     # FIX: correlated
    cat        = np.random.choice(cats, p=cat_weights)

    normal_rows.append({
        'txn_id':            _txn_id(),           # FIX: uuid
        'timestamp':         ts,
        'sender_upi':        sender,
        'receiver_upi':      receiver,
        'amount':            amt,
        'upi_app':           app,
        'merchant_category': cat,
        'sender_bank':       s_bank,
        'receiver_bank':     r_bank,
        'sender_state':      s_state,
        'receiver_state':    r_state,
        'device_id':         device_pool[sender],
        'is_new_device':     0,
        'fraud':             0,
    })

# ─── Generate fraud transactions ──────────────────────────────────────────────

print(f"Injecting {N_FRAUD:,} fraud transactions (8 patterns)...")  # IMP-02: was 6

fraud_rows = []
per_pattern = N_FRAUD // 8  # IMP-02: 375 × 8 = 3000; was N_FRAUD // 6

# Pattern 1: Late-night high-value P2P
for _ in range(per_pattern):
    sender  = random.choice(sender_pool)
    s_state = sender_home_state[sender]
    ts = _rand_ts().replace(hour=np.random.randint(1, 4),
                             minute=np.random.randint(0, 59))
    fraud_rows.append({
        'txn_id':            _txn_id(),
        'timestamp':         ts,
        'sender_upi':        sender,
        'receiver_upi':      _random_upi('mule'),
        'amount':            round(np.random.uniform(20000, 200000), 2),
        'upi_app':           np.random.choice(UPI_APPS, p=APP_WEIGHTS),
        'merchant_category': 'p2p',
        'sender_bank':       sender_home_bank[sender],
        'receiver_bank':     random.choice(BANKS),   # mule on any bank
        'sender_state':      s_state,
        'receiver_state':    random.choice([s for s in STATES if s != s_state]),  # cross-state
        'device_id':         device_pool[sender],
        'is_new_device':     0,
        'fraud':             1,
    })

# Pattern 2: Rapid velocity
for _ in range(per_pattern):
    sender  = random.choice(sender_pool[:500])
    s_state = sender_home_state[sender]
    base_ts = _rand_ts()
    ts      = base_ts + timedelta(minutes=np.random.randint(0, 30))
    fraud_rows.append({
        'txn_id':            _txn_id(),
        'timestamp':         ts,
        'sender_upi':        sender,
        'receiver_upi':      _random_upi('mule'),
        'amount':            round(np.random.uniform(500, 5000), 2),
        'upi_app':           np.random.choice(UPI_APPS, p=APP_WEIGHTS),
        'merchant_category': 'p2p',
        'sender_bank':       sender_home_bank[sender],
        'receiver_bank':     random.choice(BANKS),
        'sender_state':      s_state,
        'receiver_state':    random.choice(STATES),
        'device_id':         device_pool[sender],
        'is_new_device':     0,
        'fraud':             1,
    })

# Pattern 3: New device + high amount
for _ in range(per_pattern):
    sender  = random.choice(sender_pool)
    s_state = sender_home_state[sender]
    fraud_rows.append({
        'txn_id':            _txn_id(),
        'timestamp':         _rand_ts(),
        'sender_upi':        sender,
        'receiver_upi':      _random_upi('mule'),
        'amount':            round(np.random.uniform(30000, 150000), 2),
        'upi_app':           np.random.choice(UPI_APPS, p=APP_WEIGHTS),
        'merchant_category': 'p2p',
        'sender_bank':       sender_home_bank[sender],
        'receiver_bank':     random.choice(BANKS),
        'sender_state':      s_state,
        'receiver_state':    random.choice(STATES),
        'device_id':         _device_hash(f"new_{np.random.randint(int(1e8))}"),
        'is_new_device':     1,
        'fraud':             1,
    })

# Pattern 4: Cross-state impossible travel + new device
for _ in range(per_pattern):
    s1, s2 = random.sample(STATES, 2)
    sender  = random.choice(sender_pool)
    fraud_rows.append({
        'txn_id':            _txn_id(),
        'timestamp':         _rand_ts(),
        'sender_upi':        sender,
        'receiver_upi':      _random_upi('mule'),
        'amount':            round(np.random.uniform(5000, 80000), 2),
        'upi_app':           np.random.choice(UPI_APPS, p=APP_WEIGHTS),
        'merchant_category': 'p2p',
        'sender_bank':       random.choice(BANKS),
        'receiver_bank':     random.choice(BANKS),
        'sender_state':      s1,
        'receiver_state':    s2,
        'device_id':         _device_hash(f"travel_{np.random.randint(int(1e8))}"),
        'is_new_device':     1,
        'fraud':             1,
    })

# Pattern 5: Round-amount transfers
ROUND_AMOUNTS = [10000, 20000, 25000, 50000, 75000, 100000, 200000, 500000]
for _ in range(per_pattern):
    sender  = random.choice(sender_pool)
    s_state = sender_home_state[sender]
    fraud_rows.append({
        'txn_id':            _txn_id(),
        'timestamp':         _rand_ts(),
        'sender_upi':        sender,
        'receiver_upi':      _random_upi('mule'),
        'amount':            random.choice(ROUND_AMOUNTS),
        'upi_app':           np.random.choice(UPI_APPS, p=APP_WEIGHTS),
        'merchant_category': 'p2p',
        'sender_bank':       sender_home_bank[sender],
        'receiver_bank':     random.choice(BANKS),
        'sender_state':      s_state,
        'receiver_state':    random.choice(STATES),
        'device_id':         _device_hash(f"round_{np.random.randint(int(1e8))}"),
        'is_new_device':     np.random.choice([0, 1]),
        'fraud':             1,
    })

# Pattern 6: Dormant account sudden spike
for _ in range(per_pattern):
    sender  = random.choice(sender_pool[-2000:])
    s_state = sender_home_state[sender]
    fraud_rows.append({
        'txn_id':            _txn_id(),
        'timestamp':         _rand_ts(),
        'sender_upi':        sender,
        'receiver_upi':      _random_upi('mule'),
        'amount':            round(np.random.uniform(50000, 500000), 2),
        'upi_app':           'BHIM',
        'merchant_category': 'p2p',
        'sender_bank':       sender_home_bank[sender],
        'receiver_bank':     random.choice(BANKS),
        'sender_state':      s_state,
        'receiver_state':    random.choice(STATES),
        'device_id':         _device_hash(f"dormant_{np.random.randint(int(1e8))}"),
        'is_new_device':     1,
        'fraud':             1,
    })

# ─── IMP-02: Adversarial patterns ─────────────────────────────────────────────

# Pattern 7: Camouflaged high-value -- normal-looking amount/hour/category
# ONLY signals: is_new_device=1 + cross_state=1 + mule receiver
# Forces model to learn cross-feature interaction, not single-feature threshold
for _ in range(per_pattern):
    sender  = random.choice(sender_pool)
    s_state = sender_home_state[sender]
    r_state = random.choice([s for s in STATES if s != s_state])
    fraud_rows.append({
        'txn_id':            _txn_id(),
        'timestamp':         _rand_ts(),
        'sender_upi':        sender,
        'receiver_upi':      _random_upi('mule'),
        'amount':            round(np.random.uniform(200, 4000), 2),
        'upi_app':           np.random.choice(UPI_APPS, p=APP_WEIGHTS),
        'merchant_category': random.choice(['food', 'groceries', 'recharge', 'ecommerce']),
        'sender_bank':       sender_home_bank[sender],
        'receiver_bank':     random.choice(BANKS),
        'sender_state':      s_state,
        'receiver_state':    r_state,
        'device_id':         _device_hash(f"adv7_{np.random.randint(int(1e8))}"),
        'is_new_device':     1,
        'fraud':             1,
    })

# Pattern 8: Account takeover -- known sender, known device, behavioral shift
# Signals: amount_vs_sender_avg >> 1 + cross_state; no new device
# Hardest pattern: forces model to rely on ratio feature, not device flag
for _ in range(per_pattern):
    sender  = random.choice(sender_pool[-2000:])   # low-activity senders (low avg)
    s_state = sender_home_state[sender]
    r_state = random.choice([s for s in STATES if s != s_state])
    fraud_rows.append({
        'txn_id':            _txn_id(),
        'timestamp':         _rand_ts(),
        'sender_upi':        sender,
        'receiver_upi':      _random_upi('mule'),
        'amount':            round(np.random.uniform(40000, 300000), 2),
        'upi_app':           np.random.choice(UPI_APPS, p=APP_WEIGHTS),
        'merchant_category': 'p2p',
        'sender_bank':       sender_home_bank[sender],
        'receiver_bank':     random.choice(BANKS),
        'sender_state':      s_state,
        'receiver_state':    r_state,
        'device_id':         device_pool[sender],   # same known device -- adversarial
        'is_new_device':     0,
        'fraud':             1,
    })



print("Combining and computing velocity features...")

df = pd.DataFrame(normal_rows + fraud_rows)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Time features
df['txn_hour']   = df['timestamp'].dt.hour
df['txn_day']    = df['timestamp'].dt.dayofweek
df['is_weekend'] = (df['txn_day'] >= 5).astype(int)
df['is_night']   = ((df['txn_hour'] >= 23) | (df['txn_hour'] <= 4)).astype(int)

# Amount features
df['amount_log']      = np.log1p(df['amount'])
df['is_round_amount'] = (df['amount'] % 1000 == 0).astype(int)

# Per-sender velocity (real time-windowed count)
print("Computing per-sender velocity (slow step, ~30s)...")
df['txn_ts_unix'] = df['timestamp'].astype(np.int64) // 10**9
velocity_1h         = np.zeros(len(df), dtype=np.int32)
velocity_24h        = np.zeros(len(df), dtype=np.int32)
unique_receivers_1h = np.zeros(len(df), dtype=np.int32)   # IMP-03
amount_entropy_1h   = np.zeros(len(df), dtype=np.float64)  # IMP-03

for sender, grp in df.groupby('sender_upi'):
    idx       = grp.index.values
    times     = grp['txn_ts_unix'].values
    receivers = grp['receiver_upi'].values   # IMP-03
    amts      = grp['amount'].values          # IMP-03
    for i, (gi, t) in enumerate(zip(idx, times)):
        mask_1h  = (times[:i] >= t - 3600)  & (times[:i] < t)
        mask_24h = (times[:i] >= t - 86400) & (times[:i] < t)
        v1h  = int(mask_1h.sum())
        v24h = int(mask_24h.sum())
        velocity_1h[gi]  = v1h
        velocity_24h[gi] = v24h

        # IMP-03: unique receivers in 1h window
        recv_1h = receivers[:i][mask_1h]
        unique_receivers_1h[gi] = len(set(recv_1h))

        # IMP-03: Shannon entropy of amounts in 1h window
        amts_1h = amts[:i][mask_1h]
        if len(amts_1h) > 1:
            bins   = np.digitize(amts_1h, [100, 1000, 10000, 100000])
            counts = np.bincount(bins, minlength=6)[1:]
            probs  = counts / counts.sum()
            probs  = probs[probs > 0]
            amount_entropy_1h[gi] = float(-np.sum(probs * np.log2(probs)))
        else:
            amount_entropy_1h[gi] = 0.0

df['velocity_1h']         = velocity_1h
df['velocity_24h']        = velocity_24h
df['unique_receivers_1h'] = unique_receivers_1h  # IMP-03
df['amount_entropy_1h']   = amount_entropy_1h    # IMP-03

# IMP-03: sanity assertion -- unique_receivers can never exceed txn count
assert (df['unique_receivers_1h'] <= df['velocity_1h']).all(), \
    "unique_receivers_1h > velocity_1h -- encoding bug"

# IMP-01: Receiver graph features -- rolling 24h window per receiver
print("Computing receiver graph features (IMP-01)...")
receiver_txn_count_24h  = np.zeros(len(df), dtype=np.int32)
receiver_amount_sum_24h = np.zeros(len(df), dtype=np.float64)

for receiver, grp in df.groupby('receiver_upi'):
    idx   = grp.index.values
    times = grp['txn_ts_unix'].values
    amts  = grp['amount'].values
    for i, (gi, t) in enumerate(zip(idx, times)):
        mask = (times[:i] >= t - 86400) & (times[:i] < t)
        receiver_txn_count_24h[gi]  = int(mask.sum())
        receiver_amount_sum_24h[gi] = float(amts[:i][mask].sum())

df['receiver_txn_count_24h']  = receiver_txn_count_24h   # IMP-01
df['receiver_amount_sum_24h'] = receiver_amount_sum_24h  # IMP-01

# Amount vs sender's own historical average
sender_avg = df.groupby('sender_upi')['amount'].transform('mean')
df['amount_vs_sender_avg'] = df['amount'] / (sender_avg + 1)

# Cross-bank / cross-state flags
df['cross_bank']  = (df['sender_bank']  != df['receiver_bank']).astype(int)
df['cross_state'] = (df['sender_state'] != df['receiver_state']).astype(int)

# FIX L-05: Encode categoricals with EXPLICIT categories lists.
# Must match the lists in app.py's encode_raw_df() exactly -- otherwise
# training uses alphabetical-sort codes but inference uses list-order codes.
MERCHANT_CAT_LIST = ['p2p','food','utilities','groceries','fuel','recharge','ecommerce','rent','education']
df['upi_app_enc']      = pd.Categorical(df['upi_app'],           categories=UPI_APPS).codes
df['merchant_cat_enc'] = pd.Categorical(df['merchant_category'], categories=MERCHANT_CAT_LIST).codes
df['sender_bank_enc']  = pd.Categorical(df['sender_bank'],       categories=BANKS).codes
df['receiver_bank_enc']= pd.Categorical(df['receiver_bank'],     categories=BANKS).codes
df['sender_state_enc'] = pd.Categorical(df['sender_state'],      categories=STATES).codes
df['receiver_state_enc']= pd.Categorical(df['receiver_state'],   categories=STATES).codes

# ─── Save ─────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
df.to_csv(DATA_PATH, index=False)

# FIX 9.2: build device registry -- dict of sender_upi -> set of known device_ids
# Saved to outputs/device_registry.pkl so inference can flag new devices.
# Only includes devices seen on NORMAL transactions (known-good baseline).
# L-13 NOTE: In production, device registry would be built from ALL historical
# transactions (not filtered by fraud label), since labels are unavailable at
# deployment time. Here we use the fraud column because this is synthetic data
# and we want the registry to reflect a clean baseline.
import pickle as _pkl
normal_df = df[df['fraud'] == 0]
device_registry = (
    normal_df.groupby('sender_upi')['device_id']
    .apply(set)
    .to_dict()
)
os.makedirs('outputs', exist_ok=True)
with open('outputs/device_registry.pkl', 'wb') as _f:
    _pkl.dump(device_registry, _f)
n_senders = len(device_registry)
n_devices = sum(len(v) for v in device_registry.values())
print(f"Device registry saved -> outputs/device_registry.pkl  ({n_senders:,} senders, {n_devices:,} devices)")

# IMP-01: Save receiver_stats.csv for Option B inference lookup
# Contains precomputed 24h window stats per receiver from training data.
# encode_raw_df() loads this at inference time -- missing receiver -> 0 (new, suspicious).
receiver_stats = (
    df.groupby('receiver_upi')
    .agg(
        receiver_txn_count_24h_mean=('receiver_txn_count_24h', 'mean'),
        receiver_amount_sum_24h_mean=('receiver_amount_sum_24h', 'mean'),
        receiver_txn_count_24h_max=('receiver_txn_count_24h', 'max'),
        receiver_amount_sum_24h_max=('receiver_amount_sum_24h', 'max'),
    )
    .reset_index()
)
receiver_stats_path = os.path.join('outputs', 'receiver_stats.csv')
receiver_stats.to_csv(receiver_stats_path, index=False)
print(f"Receiver stats saved -> {receiver_stats_path}  ({len(receiver_stats):,} receivers)")

fraud_rate = df['fraud'].mean() * 100
print(f"\nDataset saved -> {DATA_PATH}")
print(f"Shape      : {df.shape}")
print(f"Fraud rate : {fraud_rate:.2f}%")

# Quick validation of new signal quality
cross_state_fraud  = df[df['fraud']==1]['cross_state'].mean()
cross_state_normal = df[df['fraud']==0]['cross_state'].mean()
print(f"\ncross_state rate  -- fraud: {cross_state_fraud:.2%}  normal: {cross_state_normal:.2%}")
cross_bank_fraud   = df[df['fraud']==1]['cross_bank'].mean()
cross_bank_normal  = df[df['fraud']==0]['cross_bank'].mean()
print(f"cross_bank rate   -- fraud: {cross_bank_fraud:.2%}   normal: {cross_bank_normal:.2%}")
