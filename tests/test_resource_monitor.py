import tempfile
import time
import unittest
from pathlib import Path

from utils.resource_monitor import (
    ResourceMonitor,
    parse_meminfo,
    parse_nvidia_smi_query,
    render_resource_status,
)


def _sample_memory():
    gib = 1024**3
    return {
        "mem_total": 16 * gib,
        "mem_available": 6 * gib,
        "mem_free": 2 * gib,
        "mem_used": 10 * gib,
        "swap_total": 2 * gib,
        "swap_free": gib,
        "swap_used": gib,
    }


class ResourceMonitorTest(unittest.TestCase):
    def test_parse_meminfo_converts_kib_to_bytes(self):
        parsed = parse_meminfo(
            "\n".join(
                [
                    "MemTotal:       1024 kB",
                    "MemAvailable:    512 kB",
                    "SwapTotal:       128 kB",
                    "Malformed:",
                ]
            )
        )

        self.assertEqual(parsed["MemTotal"], 1024 * 1024)
        self.assertEqual(parsed["MemAvailable"], 512 * 1024)
        self.assertEqual(parsed["SwapTotal"], 128 * 1024)
        self.assertNotIn("Malformed", parsed)

    def test_parse_nvidia_smi_query(self):
        parsed = parse_nvidia_smi_query(
            "0, NVIDIA GeForce RTX 5090, 4408, 32607, 97, 75, 456.90\n"
            "1, Test GPU, N/A, 24576, 0, 40, [Not Supported]\n"
        )

        self.assertEqual(parsed[0]["index"], "0")
        self.assertEqual(parsed[0]["name"], "NVIDIA GeForce RTX 5090")
        self.assertEqual(parsed[0]["memory_used_mib"], 4408.0)
        self.assertEqual(parsed[0]["power_draw_w"], 456.9)
        self.assertIsNone(parsed[1]["memory_used_mib"])
        self.assertIsNone(parsed[1]["power_draw_w"])

    def test_render_records_gpu_error_without_dropping_memory(self):
        rendered = render_resource_status(
            status="running",
            interval_seconds=1.0,
            memory=_sample_memory(),
            memory_error=None,
            gpus=None,
            gpu_error="FileNotFoundError: nvidia-smi",
            timestamp_utc="2026-06-17T00:00:00Z",
            pid=123,
            hostname="host",
        )

        self.assertIn("status: running", rendered)
        self.assertIn("memory: used=10.00GiB", rendered)
        self.assertIn("swap: used=1.00GiB", rendered)
        self.assertIn("gpu_error: FileNotFoundError: nvidia-smi", rendered)

    def test_monitor_writes_latest_file_and_stops(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sys_info" / "exp.txt"
            monitor = ResourceMonitor(
                path,
                interval_seconds=0.05,
                memory_provider=_sample_memory,
                gpu_provider=lambda: [
                    {
                        "index": "0",
                        "name": "Test GPU",
                        "memory_used_mib": 100.0,
                        "memory_total_mib": 1000.0,
                        "utilization_gpu_percent": 50.0,
                        "temperature_c": 60.0,
                        "power_draw_w": 120.5,
                    }
                ],
            )

            monitor.start()
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and not path.exists():
                time.sleep(0.01)
            self.assertTrue(path.exists())
            self.assertIn("status: running", path.read_text(encoding="utf-8"))

            monitor.stop(final_status="stopped")
            text = path.read_text(encoding="utf-8")
            self.assertIn("status: stopped", text)
            self.assertIn("gpu[0]: name=Test GPU", text)


if __name__ == "__main__":
    unittest.main()
