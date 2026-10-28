"""Bounded exact-match session cache; document text is never persisted."""
import hashlib
import json
import time

TTL_SECONDS = 300
MAX_ENTRIES = 20

def context_key(query, document_digest, history, model):
    payload = ["v2", query.strip(), document_digest, history[-5:], model]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def check_cache(cache, key, now=None):
    now = time.time() if now is None else now
    for expired in [k for k, v in cache.items() if now - v["created"] >= TTL_SECONDS]:
        del cache[expired]
    entry = cache.get(key)
    return entry["value"] if entry else None

def add_to_cache(cache, key, value, now=None):
    now = time.time() if now is None else now
    cache[key] = {"created": now, "value": value}
    while len(cache) > MAX_ENTRIES:
        del cache[min(cache, key=lambda k: cache[k]["created"])]
