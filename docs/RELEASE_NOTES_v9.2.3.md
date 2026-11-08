# Fizgig v9.2.3: the Windows installer runs again

**Fixed: `install_fizgig.bat` did nothing on Windows** (#310). A message added in 9.2.1 had brackets in it that broke the whole script, so double-clicking it stopped at once with "from was unexpected at this time", whether or not Python was installed. Thanks to @josm2026 for the report and the fix.

Updating installs weren't affected: the updater doesn't use this file. If a fresh install failed for you since 9.2.1, run `update_fizgig.bat` (it fetches the fixed installer), then `install_fizgig.bat`.
