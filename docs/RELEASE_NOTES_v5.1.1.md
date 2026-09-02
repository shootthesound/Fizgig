# Fizgig v5.1.1

## A launch splash — you see Fizgig within a second

Suggested by **[@fm3at](https://github.com/fm3at)** (#115), who put a working prototype
in the issue; **[@Davikar](https://github.com/Davikar)** confirmed the wait even on a fast
NVMe drive.

Fizgig loads a lot before its first window — on a slow drive that first launch could
take minutes with nothing on screen, and it looked hung. Now a small status window
appears within a second of launching and stays with you through the load: "Loading
modules…" while the libraries come in, then each tab as it's built, then the main window
appears fully drawn instead of blank and filling in. Nothing about the load itself
changed; it just stopped being invisible.
