"""Find out why the data servers cannot be reached: DNS, TCP to each address, HTTPS, then zarr. Sends the report to Telegram."""
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aiohttp
import fsspec
import requests
import xarray
import zarr

from src import config, notify
from src.data import check_connection

# the shot table lives on the same host (fairmast-meta/l2_shots.parquet); mastapp.site is retired (2026-10-27)
HOSTS = {
    "s3.echo.stfc.ac.uk": f"https://s3.echo.stfc.ac.uk/mast/level2/shots/{config.REFERENCE_SHOT}.zarr/zarr.json",
}
TIMEOUT_S = 10


def resolve(host):
    """Sorted list of (family, ip) for host on port 443, or [] if DNS fails."""
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return []
    return sorted({(info[0], info[4][0]) for info in infos})


def tcp_check(family, ip):
    """(ok, milliseconds or error text) for a plain TCP connection to ip:443."""
    start = time.time()
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.settimeout(TIMEOUT_S)
            sock.connect((ip, 443))
        return True, f"{1000 * (time.time() - start):.0f} ms"
    except OSError as error:
        return False, f"{type(error).__name__}: {error}"


def https_check(url):
    """(ok, short description) after fetching the first kilobyte of url."""
    try:
        with requests.get(url, stream=True, timeout=(TIMEOUT_S, 30)) as response:
            next(response.iter_content(1024), b"")
            return response.ok, f"HTTP {response.status_code}"
    except requests.RequestException as error:
        return False, f"{type(error).__name__}: {str(error)[:150]}"


def verdict(host, addresses, tcp_results, https_ok):
    """Plain-language conclusion for one host from the checks above."""
    n_ok = sum(ok for ok, _ in tcp_results)
    if not addresses:
        return f"{host}: DNS lookup failed. Check /etc/resolv.conf or the VPS provider's DNS."
    if n_ok == 0:
        return (f"{host}: no address answers on port 443. Either the server is down, or this VPS's network path "
                f"to it is blocked. Open {HOSTS[host]} in your phone's browser: if it loads there, the block is "
                f"on the VPS side: ask the provider whether outbound traffic to {addresses[0][1]} is filtered, or "
                f"try a VPS in another location.")
    if n_ok < len(addresses):
        good = [ip for (family, ip), (ok, _) in zip(addresses, tcp_results) if ok]
        return (f"{host}: only {n_ok} of {len(addresses)} addresses answer. Pin a working one by adding this "
                f"line to /etc/hosts:\n{good[0]} {host}")
    if not https_ok:
        return f"{host}: TCP works but HTTPS fails, so it is a TLS or proxy problem on the VPS."
    return f"{host}: reachable."


def check_host(host, lines):
    """Run every check for one host, append report lines, and return its verdict."""
    addresses = resolve(host)
    lines.append(f"\n{host}: {len(addresses)} addresses")
    tcp_results = []
    for family, ip in addresses:
        ok, info = tcp_check(family, ip)
        tcp_results.append((ok, info))
        lines.append(f"  tcp {ip}: {'ok' if ok else 'FAIL'} ({info})")
    https_ok, info = https_check(HOSTS[host])
    lines.append(f"  https: {'ok' if https_ok else 'FAIL'} ({info})")
    return verdict(host, addresses, tcp_results, https_ok)


def main():
    lines = [f"network diagnosis from {socket.gethostname()}",
             f"zarr {zarr.__version__}, xarray {xarray.__version__}, fsspec {fsspec.__version__}, "
             f"aiohttp {aiohttp.__version__}"]
    verdicts = [check_host(host, lines) for host in HOSTS]
    ok, message = check_connection(attempts=1)
    lines.append(f"\nzarr read of shot {config.REFERENCE_SHOT}: {'ok' if ok else 'FAIL'} ({message})")
    lines.append("\nVERDICT")
    lines += verdicts
    if all(v.endswith("reachable.") for v in verdicts) and not ok:
        lines.append("Both hosts answer but zarr cannot read the shot: send this report with your question.")
    report = "\n".join(lines)
    notify.send(report)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
