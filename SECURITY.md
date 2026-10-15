# Security scope and responsible use

This offline analysis tool does not scan networks or download arbitrary URL content. The only network-capable action is explicitly requested NASA catalog retrieval via HTTPS.

- Treat CSV catalog paths as input under the user’s control; limit files to 150MB by default.
- Do not load model.joblib files from untrusted sources. Joblib/pickle deserialization can execute code.
- The static HTML report escapes user-provided TOI and TIC strings and uses Plotly figures based on numeric results.
- Store real observations locally. Avoid committing source files that contain data subject to institutional access restrictions.
- Report technical issues through a private channel rather than posting sensitive data in a public GitHub issue.
