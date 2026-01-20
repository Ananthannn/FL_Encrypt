#!/usr/bin/env python3
import asyncio
import time
import sys
from pathlib import Path
import logging
import tempfile

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from protocol.state_machine import StateMachine
from session.master import Master
from session.worker import Worker


# ─────────────────────────────────────────────────────────────
# Logging configuration
# ─────────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s %(levelname)-5s %(name)s %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt="%Y-%m-%dT%H:%M:%S",
)

log = logging.getLogger("pqc.benchmark")
server_log = logging.getLogger("pqc.server")
transfer_log = logging.getLogger("pqc.transfer")
session_log = logging.getLogger("pqc.session")


# ─────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────

def emit_report(tester, results, total_time):
    c2s_times = [r["time_s"] for r in results["client_to_server"]]

    report = {
        "file_size_gb": tester.file_size_gb,
        "rounds": results["total_rounds"],
        "total_time_s": round(total_time, 2),
        "avg_time_s": round(sum(c2s_times) / len(c2s_times), 3),
        "best_time_s": round(min(c2s_times), 3),
        "avg_throughput_mb_s": round(
            tester.file_size_gb * 1024 / (sum(c2s_times) / len(c2s_times)),
            1,
        ),
        "disk_io": "disabled",
    }

    for k, v in report.items():
        print(f"{k:22}: {v}")


# ─────────────────────────────────────────────────────────────
# Benchmark 
# ─────────────────────────────────────────────────────────────

class ThroughputTester:
    def __init__(self, key_path: str = "./keys", file_size_gb: float = 3.0):
        self.key_path = key_path
        self.file_size_gb = file_size_gb
        self.file_size_bytes = int(file_size_gb * 1024**3)
        self.test_file = self._create_dummy_file()

    def _create_dummy_file(self) -> Path:
        test_file = Path("new_file.bin")
        if test_file.exists():
            return test_file

        log.info(
            "test_file_create size_bytes=%d",
            self.file_size_bytes,
        )

        start = time.perf_counter()

        with open(test_file, "wb") as f:
            chunk_size = 64 * 1024 * 1024
            written = 0
            while written < self.file_size_bytes:
                to_write = min(chunk_size, self.file_size_bytes - written)
                f.write(b"\x00" * to_write)
                written += to_write

        elapsed = time.perf_counter() - start

        log.info(
            "test_file_ready path=%s duration_s=%.2f",
            test_file,
            elapsed,
        )

        return test_file

    async def benchmark_roundtrip(self, iterations: int):
        results = {"client_to_server": [], "total_rounds": 0}

        server = Master(self.key_path, base_output="/tmp/dummy")
        server_task = asyncio.create_task(self._run_benchmark_server(server))

        await asyncio.sleep(3)

        try:
            for i in range(iterations):
                log.info(
                    "round_start round=%d total=%d",
                    i + 1,
                    iterations,
                )

                start = time.perf_counter()
                client = Worker(self.key_path, str(self.test_file))
                await client.connect_and_send("127.0.0.1", 8444)
                duration = time.perf_counter() - start

                throughput = (self.file_size_gb * 1024) / duration

                results["client_to_server"].append(
                    {
                        "round": i + 1,
                        "time_s": duration,
                        "throughput_mb_s": throughput,
                    }
                )

                transfer_log.info(
                    "client_to_server_complete "
                    "round=%d bytes=%d duration_s=%.3f throughput_mb_s=%.1f",
                    i + 1,
                    self.file_size_bytes,
                    duration,
                    throughput,
                )

                results["total_rounds"] += 1

        finally:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

        return results

    async def _run_benchmark_server(self, server: Worker):
        async def benchmark_handle_client(reader, writer):
            client_id = server.clients_processed
            server.clients_processed += 1

            sm = StateMachine(str(server.key_path), "server")
            server.active_clients[client_id] = (reader, writer, sm)

            try:
                await sm.transition("recv_handshake", reader=reader, writer=writer)
                await sm.transition("send_response", reader=reader, writer=writer)

                session_log.info(
                    "handshake_ok client_id=%d",
                    client_id,
                )

                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    temp_path = Path(tmp.name)

                await sm.transition(
                    "start_recv_file",
                    reader=reader,
                    writer=writer,
                    output_path=str(temp_path),
                )

                if temp_path.exists():
                    temp_path.unlink()

            except Exception as exc:
                server_log.error(
                    "client_error client_id=%d error=%s",
                    client_id,
                    exc,
                )
            finally:
                server.active_clients.pop(client_id, None)
                writer.close()
                await writer.wait_closed()

        server.handle_client = benchmark_handle_client

        server_log.info(
            "listen addr=%s port=%d",
            "127.0.0.1",
            8444,
        )

        server_instance = await asyncio.start_server(
            server.handle_client,
            "127.0.0.1",
            8444,
        )

        async with server_instance:
            await asyncio.Future()


# ─────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────

async def main():
    tester = ThroughputTester(key_path="./keys", file_size_gb=3.0)

    log.info(
        "benchmark_start file_size_gb=%.1f rounds=%d",
        tester.file_size_gb,
        10,
    )

    start = time.perf_counter()
    results = await tester.benchmark_roundtrip(iterations=10)
    total_time = time.perf_counter() - start

    emit_report(tester, results, total_time)


if __name__ == "__main__":
    asyncio.run(main())
