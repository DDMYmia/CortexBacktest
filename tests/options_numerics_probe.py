"""Data-only numerical probe; executed by the workspace verification runner."""
import json
import sys
from dataclasses import asdict
from cortexbacktest.options.pricing import black_scholes, bsm_greeks, implied_volatility

results = []
for case in json.load(sys.stdin)['cases']:
    args = [case[k] for k in ('right', 'spot', 'strike', 'time', 'rate', 'volatility', 'dividend')]
    price = black_scholes(*args)
    row = {'id': case['id'], 'price': price}
    if case['time'] > 0 and case['volatility'] > 0:
        row['greeks'] = asdict(bsm_greeks(*args))
        if case['id'].startswith(('atm-', 'dividend-', 'negative-rate-')):
            row['iv'] = implied_volatility(price, case['right'], case['spot'], case['strike'], case['time'], case['rate'], case['dividend'])
    results.append(row)
print(json.dumps(results, allow_nan=False))
