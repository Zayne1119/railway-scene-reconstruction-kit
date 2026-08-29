# Quality Gate regression fixtures

These synthetic, coordinate-free reports preserve failure patterns found during the first railway case without publishing production data:

- a release summary says `pass` while a full-length track audit says `fail`;
- canopy/support interfaces fail even though all expected objects exist;
- mesh optimization is still required despite a local release pass;
- a segment remains conditional and must not enter an authoritative merge.

The values are regression seeds, not railway-industry tolerances. Every future change to status aggregation must keep these reports blocking.
