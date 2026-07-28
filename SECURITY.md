# Security policy

Dithyramba is a local pre-alpha. Please do not use it as the sole control for
classified, regulated, clinical, legal, or otherwise high-risk material.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability that could expose
source text, filesystem paths, credentials, policy scope, or local services.
Use GitHub's private vulnerability reporting for this repository when it is
enabled. If that surface is unavailable, contact the repository owner privately
and include:

- the affected version or commit;
- the smallest reproducible case;
- expected and observed behavior;
- whether source text, local paths, authorization state, or network exposure is
  involved.

No response-time SLA is promised during pre-alpha. Confirmed issues that cross a
Library boundary, bypass policy compilation, expose a non-loopback service, or
leak protected source content receive highest priority.

## Supported versions

Only the newest tagged pre-release is considered for security fixes. Older
commits and untagged development snapshots are unsupported.

## Deployment boundary

- Keep runtime data outside the source repository and synchronized corpus roots.
- Bind the HTTP surface only to loopback.
- Do not forward the local service through a public tunnel.
- Treat corpus text and metadata as untrusted input.
- Review `AccessPolicy`, source roots, exclusions, and snapshot identity before
  sending any packet to an external provider.
