#!/usr/bin/env python3
"""Generate/validate a Kite access token for Trader_Labs only.

This is intentionally independent from the live Auto_Trader trading service:
- reads credentials from environment or an ignored lab secrets file
- writes token to Trader_Labs/intermediary_files/kite_lab_access_token.json
- does not import Auto_Trader runtime modules or live session helpers

Credential sources, in priority order:
  1. environment variables: KITE_API_KEY, KITE_API_SECRET, KITE_USER_ID,
     KITE_PASSWORD, KITE_TOTP_KEY
  2. --secrets-file path (JSON or Python assignments)
  3. Trader_Labs/secrets/kite_lab_secrets.py (gitignored)

Visible/manual browser refresh is supported for broker challenges:
  python scripts/kite_lab_token.py refresh --browser

Headless HTTP refresh is available but may fail if Zerodha presents CAPTCHA:
  python scripts/kite_lab_token.py refresh
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import onetimepass as otp
import requests
from kiteconnect import KiteConnect

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKEN_PATH = ROOT / "intermediary_files" / "kite_lab_access_token.json"
DEFAULT_SECRETS_PATH = ROOT / "secrets" / "kite_lab_secrets.py"
DEFAULT_BROWSER_PROFILE = ROOT / "intermediary_files" / "kite_lab_playwright_profile"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/136.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
API_HEADERS = {
    "Referer": "https://kite.zerodha.com/",
    "Origin": "https://kite.zerodha.com",
    "Content-Type": "application/x-www-form-urlencoded",
}


@dataclass(frozen=True)
class KiteCredentials:
    api_key: str
    api_secret: str
    user_id: str
    password: str
    totp_key: str


def mask(value: str) -> str:
    if not value:
        return ""
    return value[:2] + "***" + value[-2:]


def load_mapping_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    data: dict[str, Any] = {}
    exec(path.read_text(encoding="utf-8"), data)  # noqa: S102 - local operator-owned secrets file
    return data


def pick(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if value:
            return str(value)
    return ""


def load_credentials(secrets_file: str | None = None) -> KiteCredentials:
    file_data = load_mapping_file(Path(secrets_file).expanduser()) if secrets_file else load_mapping_file(DEFAULT_SECRETS_PATH)
    merged = dict(file_data)
    # Environment wins over file.
    for key in ["KITE_API_KEY", "KITE_API_SECRET", "KITE_USER_ID", "KITE_PASSWORD", "KITE_TOTP_KEY"]:
        if os.getenv(key):
            merged[key] = os.getenv(key)
    # Also accept names used by the live repo, without importing it.
    creds = KiteCredentials(
        api_key=pick(merged, "KITE_API_KEY", "API_KEY"),
        api_secret=pick(merged, "KITE_API_SECRET", "API_SECRET"),
        user_id=pick(merged, "KITE_USER_ID", "USER_NAME", "USER_ID"),
        password=pick(merged, "KITE_PASSWORD", "PASS", "PASSWORD"),
        totp_key=pick(merged, "KITE_TOTP_KEY", "TOTP_KEY"),
    )
    missing = [name for name, value in creds.__dict__.items() if not value]
    if missing:
        raise SystemExit(
            "Missing Kite credentials: "
            + ", ".join(missing)
            + f". Set env vars or create ignored file {DEFAULT_SECRETS_PATH}"
        )
    return creds


def read_token(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except FileNotFoundError:
        return {}


def write_token(path: Path, access_token: str, api_key: str) -> dict[str, str]:
    payload = {
        "access_token": access_token,
        "api_key_hint": mask(api_key),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "generated_at": datetime.now().astimezone().isoformat(),
        "scope": "Trader_Labs research only",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
    return payload


def token_is_today(payload: dict[str, Any]) -> bool:
    return bool(payload.get("access_token")) and str(payload.get("date")) == datetime.now().strftime("%Y-%m-%d")


def json_or_empty(response: requests.Response) -> dict[str, Any]:
    try:
        data = response.json()
        return data if isinstance(data, dict) else {}
    except ValueError:
        return {}


def response_summary(response: requests.Response, payload: dict[str, Any]) -> str:
    data = payload.get("data")
    keys = sorted(data.keys()) if isinstance(data, dict) else []
    return (
        f"status_code={response.status_code}, api_status={payload.get('status')!r}, "
        f"message={payload.get('message')!r}, data_keys={keys}, "
        f"content_type={response.headers.get('content-type', '')!r}"
    )


def find_request_token(url: str) -> str | None:
    token = parse_qs(urlparse(url).query).get("request_token")
    if token:
        return token[0]
    match = re.search(r"request_token=([A-Za-z0-9]+)", url)
    return match.group(1) if match else None


def request_token_http(creds: KiteCredentials, *, max_attempts: int = 3) -> str:
    kite = KiteConnect(api_key=creds.api_key)
    login_url = kite.login_url()
    last_summary = ""
    for attempt in range(1, max_attempts + 1):
        session = requests.Session()
        session.headers.update(BROWSER_HEADERS)
        session.get(login_url, timeout=15)
        login_response = session.post(
            "https://kite.zerodha.com/api/login",
            data={"user_id": creds.user_id, "password": creds.password},
            headers=API_HEADERS,
            timeout=15,
        )
        login_payload = json_or_empty(login_response)
        login_data = login_payload.get("data") if isinstance(login_payload.get("data"), dict) else {}
        if login_response.status_code == 200 and login_data and login_data.get("request_id"):
            twofa = session.post(
                "https://kite.zerodha.com/api/twofa",
                data={
                    "user_id": creds.user_id,
                    "request_id": login_data["request_id"],
                    "twofa_value": str(otp.get_totp(creds.totp_key)).zfill(6),
                    "twofa_type": "totp",
                    "skip_session": True,
                },
                headers=API_HEADERS,
                timeout=15,
            )
            if twofa.status_code != 200:
                raise RuntimeError(f"TOTP failed: {response_summary(twofa, json_or_empty(twofa))}")
            redirect = session.get(login_url, timeout=15, allow_redirects=True)
            token = find_request_token(redirect.url)
            if token:
                return token
            raise RuntimeError("Kite login succeeded but request_token was not present in redirect URL")
        last_summary = response_summary(login_response, login_payload)
        if "captcha" in last_summary.lower():
            raise RuntimeError(f"Kite returned CAPTCHA/challenge; use --browser. {last_summary}")
        time.sleep(min(30, 3 * attempt))
    raise RuntimeError(f"Kite HTTP login failed after {max_attempts} attempts: {last_summary}")


def request_token_browser(creds: KiteCredentials, *, headless: bool, wait_human_seconds: int) -> str:
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("Playwright is required for --browser. Install with: python -m pip install playwright && playwright install chromium") from exc

    kite = KiteConnect(api_key=creds.api_key)
    login_url = kite.login_url()
    DEFAULT_BROWSER_PROFILE.mkdir(parents=True, exist_ok=True)

    def first_visible(page, selectors: list[str], timeout_ms: int = 5000):
        deadline = time.time() + timeout_ms / 1000
        while time.time() < deadline:
            for selector in selectors:
                loc = page.locator(selector).first
                try:
                    if loc.count() and loc.is_visible(timeout=250):
                        return loc
                except Exception:
                    pass
            time.sleep(0.1)
        return None

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            str(DEFAULT_BROWSER_PROFILE),
            headless=headless,
            viewport={"width": 1365, "height": 900},
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = context.pages[0] if context.pages else context.new_page()
        page.set_default_timeout(15000)
        try:
            page.goto(login_url, wait_until="domcontentloaded")
            page.wait_for_load_state("networkidle", timeout=15000)
        except PlaywrightTimeoutError:
            pass

        token = find_request_token(page.url)
        if not token:
            user = first_visible(page, ["input#userid", "input[name='user_id']", "input[type='text']"], 12000)
            password = first_visible(page, ["input#password", "input[name='password']", "input[type='password']"], 12000)
            if not user or not password:
                context.close()
                raise RuntimeError("Could not locate Kite login fields")
            user.fill(creds.user_id)
            password.fill(creds.password)
            button = first_visible(page, ["button[type='submit']", "button:has-text('Login')", "button:has-text('Continue')"], 5000)
            if button:
                button.click()

            deadline = time.time() + wait_human_seconds
            while time.time() < deadline:
                token = find_request_token(page.url)
                if token:
                    break
                field = first_visible(
                    page,
                    ["input[name='twofa_value']", "input[name='totp']", "input[type='number']", "input[type='tel']", "input[placeholder*='TOTP']", "input[placeholder*='PIN']"],
                    1200,
                )
                if field:
                    field.fill(str(otp.get_totp(creds.totp_key)).zfill(6))
                    button = first_visible(page, ["button[type='submit']", "button:has-text('Continue')", "button:has-text('Submit')"], 5000)
                    if button:
                        button.click()
                    time.sleep(2)
                if not token:
                    try:
                        page.goto(login_url, wait_until="domcontentloaded")
                    except PlaywrightTimeoutError:
                        pass
                time.sleep(1)
        token = token or find_request_token(page.url)
        context.close()
        if not token:
            raise RuntimeError("Could not obtain request_token from browser flow")
        return token


def refresh(args: argparse.Namespace) -> dict[str, Any]:
    token_path = Path(args.token_path).expanduser()
    existing = read_token(token_path)
    if token_is_today(existing) and not args.force:
        print(json.dumps({"status": "existing", "token_path": str(token_path), "date": existing.get("date")}, indent=2))
        return existing

    creds = load_credentials(args.secrets_file)
    req_token = request_token_browser(creds, headless=args.headless, wait_human_seconds=args.wait_human_seconds) if args.browser else request_token_http(creds, max_attempts=args.max_attempts)
    kite = KiteConnect(api_key=creds.api_key)
    session = kite.generate_session(request_token=req_token, api_secret=creds.api_secret)
    payload = write_token(token_path, session["access_token"], creds.api_key)
    print(json.dumps({"status": "refreshed", "token_path": str(token_path), "date": payload["date"], "api_key_hint": payload["api_key_hint"]}, indent=2))
    return payload


def check(args: argparse.Namespace) -> None:
    creds = load_credentials(args.secrets_file)
    token_path = Path(args.token_path).expanduser()
    payload = read_token(token_path)
    if not payload.get("access_token"):
        raise SystemExit(f"No lab token found at {token_path}")
    kite = KiteConnect(api_key=creds.api_key)
    kite.set_access_token(payload["access_token"])
    profile = kite.profile()
    print(json.dumps({"status": "ok", "token_date": payload.get("date"), "user_id": profile.get("user_id"), "user_name": profile.get("user_name")}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Trader_Labs Kite token helper")
    parser.add_argument("--secrets-file", default=os.getenv("KITE_LAB_SECRETS_FILE", ""), help="Ignored JSON/Python credentials file")
    parser.add_argument("--token-path", default=os.getenv("KITE_LAB_TOKEN_PATH", str(DEFAULT_TOKEN_PATH)))
    sub = parser.add_subparsers(dest="command", required=True)

    p_refresh = sub.add_parser("refresh", help="Generate today's lab-only access token")
    p_refresh.add_argument("--force", action="store_true")
    p_refresh.add_argument("--browser", action="store_true", help="Use Playwright browser flow for broker challenges")
    p_refresh.add_argument("--headless", action="store_true", help="Run browser headless; visible is safer for challenges")
    p_refresh.add_argument("--wait-human-seconds", type=int, default=180)
    p_refresh.add_argument("--max-attempts", type=int, default=3)
    p_refresh.set_defaults(func=refresh)

    p_check = sub.add_parser("check", help="Validate current lab token with Kite profile API")
    p_check.set_defaults(func=check)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
