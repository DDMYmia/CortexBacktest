from datetime import date, datetime, time, timedelta, timezone

import polars as pl
import pytest

from cortexbacktest.options import (
    CHAIN_COLUMNS, ChainFilter, ChainQualityPolicy, ChainTimePolicy, OptionBacktestConfig,
    OptionChainSource, iter_normalized_chunks, scan_chain_file,
    OptionChainStore, QuotePolicy, attach_lifecycle, clean_option_chain,
    expiry_events_from_prices, expiry_events_from_underlying, normalize_option_chain,
    occ_symbol, option_chain_to_frames, option_contracts_from_chain, run_option_backtest,
    session_closes, validate_option_chain,
    with_underlying_prices,
)

EXPIRY = date(2026, 1, 16)
CLOSE = datetime(2026, 1, 5, 20, tzinfo=timezone.utc)


def dolthub_rows(**kw):
    """DoltHub post-no-preference/options shape: long, IV in `vol`, no spot column."""
    base = {"date": ["2026-01-05", "2026-01-05"], "act_symbol": ["spy", "spy"],
            "expiration": ["2026-01-16", "2026-01-16"], "strike": [390.0, 390.0],
            "call_put": ["Call", "Put"], "bid": [12.0, 8.0], "ask": [12.5, 8.4],
            "vol": [0.19, 0.21], "delta": [0.55, -0.45], "gamma": [0.01, 0.01],
            "theta": [-0.04, -0.05], "vega": [0.3, 0.3], "rho": [0.01, -0.01]}
    return pl.DataFrame({**base, **kw})


def optionsdx_rows():
    """OptionsDX shape: bracketed headers, one row per strike, call and put side by side."""
    return pl.DataFrame({
        "[QUOTE_DATE]": ["2026-01-05"], "[EXPIRE_DATE]": ["2026-01-16"], "[DTE]": [11.0],
        "[UNDERLYING_LAST]": [400.0], "[STRIKE]": [390.0],
        "[C_BID]": [12.0], "[C_ASK]": [12.5], "[C_IV]": [0.19], "[C_DELTA]": [0.55],
        "[P_BID]": [8.0], "[P_ASK]": [8.4], "[P_IV]": [0.21], "[P_DELTA]": [-0.45],
    })


def chain(days=(0,), spot=400.0, bid=8.0, ask=8.4, expiry=EXPIRY, strike=390.0, type_="PUT",
          underlying="SPY"):
    quote_dates = [date(2026, 1, 5) + timedelta(days=d) for d in days]
    spots = spot if isinstance(spot, list) else [spot] * len(quote_dates)
    frame = pl.DataFrame({"date": quote_dates, "symbol": [underlying] * len(quote_dates),
                          "underlying_last": spots, "expiration": [expiry] * len(quote_dates),
                          "strike": [strike] * len(quote_dates),
                          "type": [type_] * len(quote_dates),
                          "bid": [bid] * len(quote_dates), "ask": [ask] * len(quote_dates)})
    return normalize_option_chain(frame, "long")


def test_dolthub_long_file_normalizes_to_the_canonical_schema():
    result = normalize_option_chain(dolthub_rows(), "dolthub")
    assert result.columns == list(CHAIN_COLUMNS)
    assert result["right"].to_list() == ["CALL", "PUT"]
    assert result["underlying"].to_list() == ["SPY", "SPY"]           # lower-case input
    assert result["implied_volatility"].to_list() == [0.19, 0.21]     # `vol` is IV here
    assert result["dte"].to_list() == [11, 11]
    assert result["mid"].to_list() == [12.25, 8.2]
    assert result["option_symbol"].to_list() == ["SPY   260116C00390000",
                                                 "SPY   260116P00390000"]
    assert result["quote_date"].dtype == pl.Datetime("us", "UTC")
    assert result["quote_date"][0] == CLOSE
    assert result["expiry"][0] == datetime(2026, 1, 16, 20, tzinfo=timezone.utc)
    assert result["underlying_price"].null_count() == 2                # DoltHub ships no spot


def test_time_policy_places_dates_and_keeps_a_real_time_of_day():
    policy = ChainTimePolicy(quote_time=time(19, 45), expiry_time=time(13, 30))
    result = normalize_option_chain(dolthub_rows(), "dolthub", time_policy=policy)
    assert result["quote_date"][0] == datetime(2026, 1, 5, 19, 45, tzinfo=timezone.utc)
    assert result["expiry"][0] == datetime(2026, 1, 16, 13, 30, tzinfo=timezone.utc)

    intraday = dolthub_rows(date=["2026-01-05 15:31:00"] * 2)
    assert normalize_option_chain(intraday, "dolthub", time_policy=policy)["quote_date"][0] == (
        datetime(2026, 1, 5, 15, 31, tzinfo=timezone.utc))

    aware = dolthub_rows(date=[datetime(2026, 1, 5, 11, tzinfo=timezone(timedelta(hours=-5)))] * 2)
    assert normalize_option_chain(aware, "dolthub")["quote_date"][0] == (
        datetime(2026, 1, 5, 16, tzinfo=timezone.utc))


def test_a_named_zone_converts_each_date_on_its_own_side_of_daylight_saving():
    """A US close is 16:00 New York, which is 20:00 UTC in summer and 21:00 in
    winter. A sample spanning years must not pin one UTC time of day."""
    rows = pl.DataFrame({"date": ["2020-01-02", "2020-07-01"], "symbol": ["SPY"] * 2,
                         "expiration": ["2020-01-17", "2020-07-17"], "strike": [320.0, 310.0],
                         "type": ["PUT"] * 2, "bid": [1.0, 1.0], "ask": [1.2, 1.2],
                         "underlying_last": [320.0, 310.0]})
    policy = ChainTimePolicy(quote_time=time(16, 0), expiry_time=time(16, 0),
                             zone="America/New_York")
    result = normalize_option_chain(rows, "long", time_policy=policy).sort("quote_date")

    assert result["quote_date"].to_list() == [
        datetime(2020, 1, 2, 21, tzinfo=timezone.utc),      # EST
        datetime(2020, 7, 1, 20, tzinfo=timezone.utc)]      # EDT
    assert result["expiry"].to_list() == [
        datetime(2020, 1, 17, 21, tzinfo=timezone.utc),
        datetime(2020, 7, 17, 20, tzinfo=timezone.utc)]
    # Without a zone the same times are read as UTC, an hour off for half the year.
    naive = normalize_option_chain(rows, "long", time_policy=ChainTimePolicy(
        quote_time=time(16, 0), expiry_time=time(16, 0))).sort("quote_date")
    assert naive["quote_date"][0] == datetime(2020, 1, 2, 16, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="unknown time zone"):
        ChainTimePolicy(zone="Mars/Olympus")
    with pytest.raises(ValueError, match="name a zone"):
        ChainTimePolicy(quote_time=time(16, 0, tzinfo=timezone.utc))


def test_optionsdx_wide_file_splits_into_call_and_put_rows():
    result = normalize_option_chain(optionsdx_rows(), "optionsdx", underlying="SPY")
    assert result["right"].to_list() == ["CALL", "PUT"]
    assert result["bid"].to_list() == [12.0, 8.0]
    assert result["implied_volatility"].to_list() == [0.19, 0.21]
    assert result["delta"].to_list() == [0.55, -0.45]
    assert result["underlying_price"].to_list() == [400.0, 400.0]
    assert result["dte"].to_list() == [11, 11]           # recomputed, not the vendor's column
    with pytest.raises(ValueError, match="carries no ticker column"):
        normalize_option_chain(optionsdx_rows(), "optionsdx")
    with pytest.raises(ValueError, match="per leg"):
        normalize_option_chain(optionsdx_rows(), "optionsdx", underlying="SPY",
                               overrides={"bid": "[C_BID]"})


def test_kaggle_spy_parquet_is_a_wide_file_with_a_known_ticker():
    """The Kaggle SPY EOD export ships OptionsDX's wide layout, SPY-only, no ticker column.

    Column names are the ones `dudesurfin/spy-options-eod-volatility-surface-2010-2023`
    actually carries, padding and brackets included, plus the columns this layer
    ignores (`[C_SIZE]` is a string like "13 x 5", not a size we may use).
    """
    raw = pl.DataFrame({
        " [QUOTE_UNIXTIME]": [1577995200], "[QUOTE_READTIME]": ["2020-01-02 16:00"],
        "[QUOTE_DATE]": ["2020-01-02"], "[UNDERLYING_LAST]": [324.87],
        "[EXPIRE_DATE]": ["2020-01-17"], "[DTE]": [15.0], "[STRIKE]": [320.0],
        "[C_SIZE]": ["13 x 5"], "[C_BID]": [6.2], "[C_ASK]": [6.4], "[C_IV]": [0.128],
        "[C_DELTA]": [0.72], "[C_GAMMA]": [0.03], "[C_VEGA]": [0.28], "[C_THETA]": [-0.09],
        "[C_RHO]": [0.09], "[C_VOLUME]": [1201.0],
        "[P_SIZE]": ["8 x 21"], "[P_BID]": [1.7], "[P_ASK]": [1.8], "[P_IV]": [0.141],
        "[P_DELTA]": [-0.28], "[P_GAMMA]": [0.03], "[P_VEGA]": [0.28], "[P_THETA]": [-0.08],
        "[P_RHO]": [-0.05], "[P_VOLUME]": [3402.0],
    })
    result = normalize_option_chain(raw, "kaggle_spy_eod")
    assert result["underlying"].to_list() == ["SPY", "SPY"]       # from the source default
    assert result["option_symbol"].to_list() == ["SPY   200117C00320000",
                                                 "SPY   200117P00320000"]
    assert result["underlying_price"].to_list() == [324.87, 324.87]
    assert result["implied_volatility"].to_list() == [0.128, 0.141]
    assert result["theta"].to_list() == [-0.09, -0.08]
    assert result["volume"].to_list() == [1201.0, 3402.0]
    assert result["dte"].to_list() == [15, 15]
    assert validate_option_chain(result).ok
    # A different ticker (an OptionsDX file of the same shape) still overrides it.
    assert normalize_option_chain(raw, "kaggle_spy_eod",
                                  underlying="XSP")["underlying"].to_list() == ["XSP", "XSP"]


def test_unknown_headers_need_overrides_or_a_custom_source():
    odd = dolthub_rows().rename({"bid": "bid_px", "ask": "ask_px"})
    with pytest.raises(ValueError, match="missing required columns"):
        normalize_option_chain(odd, "dolthub")
    result = normalize_option_chain(odd, "dolthub", overrides={"bid": "bid_px", "ask": "ask_px"})
    assert result["bid"].to_list() == [12.0, 8.0]
    with pytest.raises(ValueError, match="unknown fields"):
        normalize_option_chain(odd, "dolthub", overrides={"bid_price": "bid_px"})
    with pytest.raises(ValueError, match="unknown chain source"):
        normalize_option_chain(odd, "nope")

    custom = OptionChainSource("house", "long",
                               {**{k: (k,) for k in ("strike", "bid", "ask")},
                                "quote_date": ("as_of",), "underlying": ("und",),
                                "expiry": ("exp",), "right": ("cp",)})
    house = pl.DataFrame({"as_of": ["2026-01-05"], "und": ["SPY"], "exp": ["2026-01-16"],
                          "strike": [390.0], "cp": ["P"], "bid": [8.0], "ask": [8.4]})
    assert normalize_option_chain(house, custom)["right"].to_list() == ["PUT"]


def test_unrecognised_option_type_is_an_error_not_a_silent_null():
    with pytest.raises(ValueError, match="unrecognised option type"):
        normalize_option_chain(dolthub_rows(call_put=["Call", "Straddle"]), "dolthub")


def test_occ_symbols_are_padded_to_the_21_character_standard():
    assert occ_symbol("spy", date(2022, 7, 15), "PUT", 380) == "SPY   220715P00380000"
    assert occ_symbol("BRK.B", datetime(2026, 1, 16, 20, tzinfo=timezone.utc), "CALL", 412.5) == (
        "BRK.B 260116C00412500")
    assert len(occ_symbol("SPY", EXPIRY, "CALL", 5.25)) == 21
    with pytest.raises(ValueError, match="not representable"):
        occ_symbol("SPY", EXPIRY, "CALL", 100_000.0)


def test_scalar_and_vectorised_occ_symbols_agree():
    """`occ_symbol` is the readable rule; normalisation uses a Polars expression
    for speed. Two implementations of one rule stay honest only if pinned."""
    tickers, expiries = ["SPY", "brk.b", "F"], [date(2020, 1, 17), date(2026, 12, 31)]
    strikes = [0.5, 5.25, 100.0, 412.5, 99999.999]
    rows = {"date": [], "symbol": [], "expiration": [], "strike": [], "type": [],
            "bid": [], "ask": [], "underlying_last": []}
    expected = []
    for ticker in tickers:
        for expiry in expiries:
            for strike in strikes:
                for right in ("CALL", "PUT"):
                    rows["date"].append(date(2020, 1, 2))
                    rows["symbol"].append(ticker)
                    rows["expiration"].append(expiry)
                    rows["strike"].append(strike)
                    rows["type"].append(right)
                    rows["bid"].append(1.0)
                    rows["ask"].append(1.2)
                    rows["underlying_last"].append(100.0)
                    expected.append(occ_symbol(ticker, expiry, right, strike))
    chain = normalize_option_chain(pl.DataFrame(rows), "long")
    assert sorted(chain["option_symbol"].to_list()) == sorted(expected)
    assert all(len(symbol) >= 21 for symbol in expected)


def test_underlying_prices_are_joined_by_session_date():
    prices = pl.DataFrame({"date": [date(2026, 1, 5), date(2026, 1, 6)],
                           "symbol": ["spy", "spy"], "close": [400.0, 402.0]})
    joined = with_underlying_prices(normalize_option_chain(dolthub_rows(), "dolthub"), prices)
    assert joined["underlying_price"].to_list() == [400.0, 400.0]
    assert joined.columns == normalize_option_chain(dolthub_rows(), "dolthub").columns
    with pytest.raises(ValueError, match="missing columns"):
        with_underlying_prices(joined, prices.drop("close"))


def test_validation_separates_structural_errors_from_quote_defects():
    good = chain()
    assert validate_option_chain(good).ok

    crossed = good.with_columns(pl.lit(7.0).alias("ask"))
    report = validate_option_chain(crossed)
    assert not report.ok and any("crossed" in e for e in report.errors)

    duplicated = pl.concat([good, good])
    assert any("duplicate" in e for e in validate_option_chain(duplicated).errors)

    backwards = normalize_option_chain(
        pl.DataFrame({"date": ["2026-01-20"], "symbol": ["SPY"], "expiration": ["2026-01-16"],
                      "strike": [390.0], "type": ["PUT"], "bid": [8.0], "ask": [8.4]}), "long")
    assert any("before the quote date" in e for e in validate_option_chain(backwards).errors)

    defects = good.with_columns(pl.lit(0.0).alias("bid"), pl.lit(9.0).alias("implied_volatility"),
                                pl.lit(0.4).alias("delta"))
    warnings = validate_option_chain(defects).warnings
    assert validate_option_chain(defects).ok
    assert any("zero-bid" in w for w in warnings)
    assert any("wider than" in w for w in warnings)
    assert any("implied volatilities outside" in w for w in warnings)
    assert any("wrong-signed" in w for w in warnings)          # a positive-delta put
    assert any("without a positive underlying price" in w
               for w in validate_option_chain(chain().with_columns(
                   pl.lit(None, pl.Float64).alias("underlying_price"))).warnings)


def test_values_the_quote_type_rejects_are_errors_not_warnings():
    """Found on real data: 217k of the Kaggle SPY rows carry a negative IV.

    `OptionQuote` refuses those outright, so a chain holding them cannot become
    frames at all. Validation has to name the count rather than let the adapter
    fail one row deep inside a dataclass constructor.
    """
    good = chain()
    for column, value in (("implied_volatility", -0.2), ("open_interest", -5.0),
                          ("delta", -1.4)):
        broken = good.with_columns(pl.lit(value).alias(column))
        report = validate_option_chain(broken)
        assert not report.ok, column
        with pytest.raises(ValueError):
            option_chain_to_frames(broken)
        cleaned, removed = clean_option_chain(broken)
        assert removed == {"unrepresentable_feature": 1}
        assert cleaned.is_empty()

    # A wrong-signed but possible delta stays a warning: it is suspicious data,
    # not data the engine cannot hold.
    suspicious = good.with_columns(pl.lit(0.4).alias("delta"))
    assert validate_option_chain(suspicious).ok
    assert any("wrong-signed" in w for w in validate_option_chain(suspicious).warnings)


def test_put_call_parity_screen_flags_an_inconsistent_pair():
    consistent = pl.concat([chain(type_="PUT", bid=8.0, ask=8.4),
                            chain(type_="CALL", bid=18.0, ask=18.4)])   # C-P == S-K == 10
    # Not `no warnings`: this fixture publishes no open interest, which the
    # validator says so out loud. What is asserted here is the parity screen.
    assert not any("parity" in w for w in validate_option_chain(consistent).warnings)
    broken = pl.concat([chain(type_="PUT", bid=8.0, ask=8.4),
                        chain(type_="CALL", bid=60.0, ask=60.4)])
    assert any("parity" in w for w in validate_option_chain(broken).warnings)
    # A tolerance wide enough to cover the break silences it; the screen is a
    # judgement call, not a fact about the data.
    assert not any("parity" in w for w in validate_option_chain(
        broken, ChainQualityPolicy(parity_tolerance=1.0)).warnings)


def test_cleaning_reports_which_rule_removed_each_row():
    rows = pl.concat([
        chain(bid=8.0, ask=8.4),                            # keeper
        chain(strike=380.0, bid=0.0, ask=0.4),              # zero bid
        chain(strike=370.0, bid=1.0, ask=9.0),              # wide spread
        chain(strike=360.0, bid=9.0, ask=1.0),              # crossed
        chain(strike=350.0, bid=1.0, ask=1.2).with_columns(
            pl.lit(9.0).alias("implied_volatility")),       # IV outlier
    ])
    cleaned, removed = clean_option_chain(pl.concat([rows, rows.head(1)]))
    assert cleaned["strike"].to_list() == [390.0]
    assert removed == {"crossed": 1, "zero_bid": 1, "wide_spread": 1,
                       "implied_volatility": 1, "duplicate": 1}

    kept, counts = clean_option_chain(rows, ChainFilter(
        drop_zero_bid=False, max_relative_spread=None, implied_volatility_bounds=None))
    assert sorted(kept["strike"].to_list()) == [350.0, 370.0, 380.0, 390.0]
    assert counts == {"crossed": 1}


def test_cleaning_filters_liquidity_and_dte_windows():
    rows = pl.concat([chain(days=(0,)), chain(days=(1,), strike=380.0)]).with_columns(
        pl.Series("open_interest", [10.0, 900.0]), pl.Series("volume", [0.0, 5.0]))
    assert clean_option_chain(rows, ChainFilter(min_open_interest=100))[0].height == 1
    assert clean_option_chain(rows, ChainFilter(min_volume=1))[0]["strike"].to_list() == [380.0]
    assert clean_option_chain(rows, ChainFilter(dte_range=(11, 11)))[0].height == 1
    with pytest.raises(ValueError, match="inverted"):
        ChainFilter(dte_range=(30, 10))


def test_store_partitions_by_ticker_and_year_and_prunes_on_scan(tmp_path):
    store = OptionChainStore(tmp_path)
    spy = chain(days=(0, 1))
    qqq = normalize_option_chain(
        pl.DataFrame({"date": ["2027-02-01"], "symbol": ["QQQ"], "expiration": ["2027-02-19"],
                      "strike": [500.0], "type": ["CALL"], "bid": [3.0], "ask": [3.2],
                      "underlying_last": [505.0]}), "long")
    assert store.scan().collect().is_empty()
    written = store.write(pl.concat([spy, qqq]))
    assert sorted(str(p.relative_to(tmp_path).parent) for p in written) == [
        "underlying=QQQ/year=2027", "underlying=SPY/year=2026"]

    everything = store.scan().collect()
    assert everything.columns == spy.columns
    assert everything.height == 3
    assert everything["underlying"].to_list() == ["SPY", "SPY", "QQQ"]
    assert everything.filter(pl.col("underlying") == "SPY").equals(spy)
    assert store.scan(underlyings=["spy"]).collect().height == 2
    assert store.scan(start=datetime(2026, 1, 6, tzinfo=timezone.utc)).collect().height == 2
    assert store.scan(underlyings=["SPY"],
                      end=datetime(2026, 1, 5, 20, tzinfo=timezone.utc)).collect().height == 1

    store.write(qqq)                                    # appends, never rewrites
    assert len(list(tmp_path.rglob("*.parquet"))) == 3
    with pytest.raises(ValueError, match="crossed"):
        store.write(spy.with_columns(pl.lit(1.0).alias("ask")))


def test_frames_carry_an_underlying_quote_and_the_publication_lag():
    policy = ChainTimePolicy(publication_lag=timedelta(minutes=15))
    frames = option_chain_to_frames(chain(days=(0, 1)), time_policy=policy,
                                    underlying_half_spread=0.01)
    assert [f.timestamp for f in frames] == [CLOSE + timedelta(minutes=15),
                                             CLOSE + timedelta(days=1, minutes=15)]
    quotes = frames[0].quotes
    assert set(quotes) == {"SPY", "SPY   260116P00390000"}
    assert quotes["SPY"].bid == pytest.approx(399.99) and quotes["SPY"].ask == pytest.approx(400.01)
    assert quotes["SPY"].event_time == CLOSE and quotes["SPY"].available_at == frames[0].timestamp
    option = quotes["SPY   260116P00390000"]
    assert (option.bid, option.ask) == (8.0, 8.4)
    assert option.bid_size is None and option.ask_size is None      # EOD files publish no size


def test_frames_refuse_to_invent_an_underlying_mark():
    without_spot = chain().with_columns(pl.lit(None, pl.Float64).alias("underlying_price"))
    with pytest.raises(ValueError, match="needs an underlying price"):
        option_chain_to_frames(without_spot)
    assert set(option_chain_to_frames(without_spot, include_underlying=False)[0].quotes) == {
        "SPY   260116P00390000"}
    disagreeing = pl.concat([chain(), chain(strike=380.0, spot=401.0)])
    with pytest.raises(ValueError, match="disagrees"):
        option_chain_to_frames(disagreeing)
    with pytest.raises(ValueError, match="crossed"):
        option_chain_to_frames(chain(bid=9.0, ask=1.0))


def test_contract_defaults_are_listed_equity_options_and_stay_overridable():
    contracts = option_contracts_from_chain(chain())
    assert len(contracts) == 1
    contract = contracts[0]
    assert (contract.symbol, contract.underlying, contract.right) == (
        "SPY   260116P00390000", "SPY", "PUT")
    assert (contract.exercise_style, contract.settlement, contract.multiplier) == (
        "american", "physical", 100)
    assert contract.expiry == datetime(2026, 1, 16, 20, tzinfo=timezone.utc)
    index_style = option_contracts_from_chain(chain(underlying="SPX"), exercise_style="european",
                                              settlement="cash")[0]
    assert (index_style.exercise_style, index_style.settlement) == ("european", "cash")


def test_expiry_events_come_from_the_chain_and_are_skipped_when_absent():
    rows = chain(days=(0, 11), spot=[400.0, 395.0])
    contracts = option_contracts_from_chain(rows)
    events = expiry_events_from_underlying(rows, contracts)
    assert len(events) == 1
    assert events[0].underlying_price == 395.0 and events[0].quantity is None
    assert events[0].event_time == datetime(2026, 1, 16, 20, tzinfo=timezone.utc)
    assert expiry_events_from_underlying(chain(days=(0,)), contracts) == ()

    frames = option_chain_to_frames(rows)
    attached = attach_lifecycle(frames, events)
    assert [len(f.lifecycle) for f in attached] == [0, 1]
    with pytest.raises(ValueError, match="no frame at or after"):
        attach_lifecycle(frames[:1], events)


def test_a_normalised_chain_runs_end_to_end_through_the_option_backtest():
    """The point of the module: vendor rows in, engine result out, no hand-built quotes."""
    raw = pl.DataFrame({
        "date": ["2026-01-05", "2026-01-06", "2026-01-16"],
        "act_symbol": ["SPY"] * 3, "expiration": ["2026-01-16"] * 3,
        "strike": [390.0] * 3, "call_put": ["Put"] * 3,
        "bid": [8.0, 7.0, 0.0], "ask": [8.4, 7.4, 0.05], "vol": [0.21, 0.2, 0.15],
    })
    prices = pl.DataFrame({"date": [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 16)],
                           "symbol": ["SPY"] * 3, "close": [400.0, 401.0, 395.0]})
    normalized = with_underlying_prices(normalize_option_chain(raw, "dolthub"), prices)
    cleaned, removed = clean_option_chain(
        normalized, ChainFilter(drop_zero_bid=False, max_relative_spread=None))
    assert removed == {} and cleaned.height == 3

    contracts = option_contracts_from_chain(cleaned)
    frames = attach_lifecycle(option_chain_to_frames(cleaned),
                              expiry_events_from_underlying(cleaned, contracts))
    symbol = contracts[0].symbol
    targets = {frames[0].timestamp: {symbol: -1}}
    result = run_option_backtest(
        contracts, frames, lambda frame, account: targets.get(frame.timestamp, {}),
        config=OptionBacktestConfig(require_sizes=False, option_fee=0.65,
                                    quote_policy=QuotePolicy(allow_zero_bid=True)))
    assert result.fills["price"].to_list() == [7.0]          # next frame's bid, sold
    assert result.summary()["net_pnl"] == pytest.approx(700 - 0.65)
    assert result.lifecycle["kind"].to_list() == ["expiry"]     # 395 > 390, worthless
    assert result.lifecycle["fixing"].to_list() == [395.0]      # the chain's own close


def _wide_file(tmp_path, sessions=6, strikes=3, suffix=".parquet"):
    """A small Kaggle-shaped file spanning several sessions."""
    rows = {"[QUOTE_DATE]": [], "[EXPIRE_DATE]": [], "[UNDERLYING_LAST]": [], "[STRIKE]": [],
            "[C_BID]": [], "[C_ASK]": [], "[P_BID]": [], "[P_ASK]": []}
    for session in range(sessions):
        for strike in range(strikes):
            rows["[QUOTE_DATE]"].append((date(2020, 1, 2) + timedelta(days=session)).isoformat())
            rows["[EXPIRE_DATE]"].append("2020-02-21")
            rows["[UNDERLYING_LAST]"].append(320.0 + session)
            rows["[STRIKE]"].append(310.0 + strike)
            rows["[C_BID]"].append(11.0 + strike)
            rows["[C_ASK]"].append(11.2 + strike)
            rows["[P_BID]"].append(1.0 + strike)
            rows["[P_ASK]"].append(1.2 + strike)
    frame = pl.DataFrame(rows)
    path = tmp_path / f"spy_eod_2020{suffix}"
    frame.write_parquet(path) if suffix == ".parquet" else frame.write_csv(path)
    return path


def test_chunked_normalisation_matches_reading_the_whole_file(tmp_path):
    """The bounded-memory path must be a pure memory optimisation, not a variant.

    A Kaggle year does not fit comfortably in memory, so ingestion normalises a
    few sessions at a time; concatenating those chunks has to be exactly the
    frame a single-shot normalise would have produced.
    """
    path = _wide_file(tmp_path)
    whole = normalize_option_chain(pl.read_parquet(path), "kaggle_spy_eod")
    chunks = list(iter_normalized_chunks(path, "kaggle_spy_eod", dates_per_chunk=2))

    assert len(chunks) == 3                                   # 6 sessions, 2 at a time
    assert pl.concat(chunks).sort(["quote_date", "underlying", "expiry",
                                   "strike", "right"]).equals(whole)
    assert all(chunk.columns == list(CHAIN_COLUMNS) for chunk in chunks)


def test_chunking_never_splits_a_session_across_chunks(tmp_path):
    """A half-delivered cross-section would break parity checks and frames."""
    path = _wide_file(tmp_path, sessions=5)
    chunks = list(iter_normalized_chunks(path, "kaggle_spy_eod", dates_per_chunk=2))

    seen: set = set()
    for chunk in chunks:
        sessions = set(chunk["quote_date"].dt.date().to_list())
        assert not sessions & seen                            # no session appears twice
        seen |= sessions
        for session in sessions:
            rows = chunk.filter(pl.col("quote_date").dt.date() == session)
            assert rows.height == 6                           # 3 strikes x call/put, intact
    assert len(seen) == 5


def test_chunked_normalisation_reads_csv_and_validates_its_window(tmp_path):
    path = _wide_file(tmp_path, sessions=2, suffix=".txt")
    assert scan_chain_file(path).collect().height == 6
    chunks = list(iter_normalized_chunks(path, "kaggle_spy_eod", dates_per_chunk=10))
    assert len(chunks) == 1 and chunks[0].height == 12

    with pytest.raises(ValueError, match="dates_per_chunk"):
        list(iter_normalized_chunks(path, "kaggle_spy_eod", dates_per_chunk=0))
    with pytest.raises(ValueError, match="no quote-date column"):
        list(iter_normalized_chunks(pl.LazyFrame({"nope": [1]}), "kaggle_spy_eod"))


class TestSettlementFixingsFromAnIndependentSource:
    """The fixing does not have to come from the chain being backtested.

    A chain's own `underlying_price` has one vendor marking both sides of the
    trade. An independently sourced close at least disagrees when one of them is
    wrong: against the Kaggle SPY chain, a minute-bar tape agreed to a
    mean of 9.5 cents over December 2023, which is evidence rather than hope.
    """

    def test_a_supplied_close_is_what_settles(self):
        contracts = option_contracts_from_chain(chain(days=(0, 11)))
        events = expiry_events_from_prices({("SPY", EXPIRY): 405.25}, contracts)
        assert [(event.symbol, event.underlying_price) for event in events] == [
            (contracts[0].symbol, 405.25)]

    def test_a_contract_with_no_fixing_gets_no_event(self):
        # Silence here is what makes the engine fail loudly later rather than
        # expiring a held position at an invented price.
        assert expiry_events_from_prices({}, option_contracts_from_chain(chain())) == ()

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
    def test_a_fixing_that_is_not_a_price_is_refused(self, bad):
        contracts = option_contracts_from_chain(chain())
        with pytest.raises(ValueError, match="not a price"):
            expiry_events_from_prices({("SPY", EXPIRY): bad}, contracts)

    def test_session_closes_takes_the_close_not_a_later_print(self):
        # A minute tape runs past 16:00 ET; the 16:05 print is not the close.
        stamps = [datetime(2026, 1, 5, hour, minute, tzinfo=timezone.utc)
                  for hour, minute in ((20, 59), (21, 0), (21, 5))]
        tape = pl.DataFrame({"timestamp": stamps, "close": [399.0, 400.25, 401.8],
                             "ticker": ["SPY"] * 3})
        assert session_closes(tape) == {("SPY", date(2026, 1, 5)): 400.25}

    def test_the_two_fixing_sources_agree_on_the_same_chain(self):
        # `expiry_events_from_underlying` is now this function fed from the
        # chain, so the two must not drift apart.
        data = chain(days=(0, 11), spot=[400.0, 405.25])
        contracts = option_contracts_from_chain(data)
        assert expiry_events_from_underlying(data, contracts) == expiry_events_from_prices(
            {("SPY", EXPIRY): 405.25}, contracts)


def test_a_space_padded_vendor_csv_keeps_its_numbers():
    """OptionsDX writes a space after every comma; the cast must not eat them.

    `cast(Float64, strict=False)` turns ` 323.59` into null and says nothing, so
    an unstripped ingest lands every price, strike and Greek as null and fails
    somewhere far away from the cause. The Kaggle QQQ 2020-2022 file
    (1,775,749 rows) is written exactly this way.
    """
    padded = optionsdx_rows().select(
        [pl.col(c).cast(pl.String).map_elements(lambda v: f" {v}", return_dtype=pl.String)
         .alias(f" {c}") for c in optionsdx_rows().columns])
    result = normalize_option_chain(padded, "optionsdx", underlying="QQQ")
    assert result.height == 2
    assert result["underlying_price"].to_list() == [400.0, 400.0]
    assert sorted(result["bid"].to_list()) == [8.0, 12.0]
    assert result["quote_date"].dt.date().to_list() == [date(2026, 1, 5)] * 2
    assert result["strike"].null_count() == 0
