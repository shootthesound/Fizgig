"""onnxruntime telemetry off. ORT_DISABLE_TELEMETRY=1 (set in fizgig/__init__.py) covers the Linux builds; the
Windows builds log through their own switch, turned off here. Call before a session or InsightFace loads."""


def off():
    try:
        import onnxruntime
        onnxruntime.disable_telemetry_events()
    except Exception:
        pass
